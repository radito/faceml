#[cfg(target_os = "macos")]
mod macos {
    use std::cell::OnceCell;
    use std::collections::HashMap;
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant};

    use block2::RcBlock;
    use dispatch2::{DispatchQueue, DispatchQueueAttr};
    use facefeature::detector::apple_vision::AppleVisionDetector;
    use facefeature::face_id::{
        FaceCaptureStatus, FaceCaptureTarget, FaceIdClient, FaceIdDatabaseMode, FaceIdentityMatch,
        aligned_sface_tensor_bgra, benchmark_sface_coreml, landmark_pitch_degrees,
    };
    use facefeature::{FaceTracker, LandmarkKind, Point, TrackedFace};
    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, Bool, ProtocolObject};
    use objc2::{AnyThread, DefinedClass, MainThreadOnly, define_class, msg_send};
    use objc2_app_kit::{
        NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSBackingStoreType,
        NSWindow, NSWindowDelegate, NSWindowStyleMask,
    };
    use objc2_av_foundation::{
        AVAuthorizationStatus, AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceInput,
        AVCaptureInput, AVCaptureOutput, AVCaptureSession, AVCaptureSessionPreset1280x720,
        AVCaptureVideoDataOutput, AVCaptureVideoDataOutputSampleBufferDelegate,
        AVCaptureVideoPreviewLayer, AVLayerVideoGravityResizeAspectFill, AVMediaTypeVideo,
    };
    use objc2_core_foundation::{CFString, CGPoint, CGRect, CGSize};
    use objc2_core_graphics::{CGColor, CGMutablePath};
    use objc2_core_media::CMSampleBuffer;
    use objc2_core_video::{
        CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
        CVPixelBufferGetHeight, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
        CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelBufferPixelFormatTypeKey,
        kCVPixelFormatType_32BGRA, kCVReturnSuccess,
    };
    use objc2_foundation::{
        MainThreadMarker, NSDictionary, NSNotification, NSNumber, NSObject, NSObjectProtocol,
        NSString, ns_string,
    };
    use objc2_quartz_core::{
        CAShapeLayer, CATextLayer, CATransaction, kCALineCapRound, kCALineJoinRound,
    };

    const WINDOW_WIDTH: f64 = 960.0;
    const WINDOW_HEIGHT: f64 = 640.0;
    const FRAME_INTERVAL: Duration = Duration::from_millis(33);
    const LABEL_INTERVAL: Duration = Duration::from_millis(200);
    const PRESENTATION_DELAY_SECONDS: f64 = 1.0 / 60.0;

    #[derive(Debug)]
    struct FrameDelegateIvars {
        preview_layer: usize,
        overlay_layer: usize,
        last_processed: Mutex<Option<Instant>>,
        last_label_refresh: Mutex<Option<Instant>>,
        tracker: Mutex<FaceTracker>,
        face_id: Option<Arc<FaceIdClient>>,
        reported_error: AtomicBool,
    }

    define_class!(
        // SAFETY: NSObject has no subclassing requirements, and all ivars are thread-safe.
        #[unsafe(super = NSObject)]
        #[thread_kind = AnyThread]
        #[ivars = FrameDelegateIvars]
        #[derive(Debug)]
        struct FrameDelegate;

        // SAFETY: NSObjectProtocol has no additional safety requirements.
        unsafe impl NSObjectProtocol for FrameDelegate {}

        // SAFETY: The callback signature matches AVFoundation's delegate protocol.
        unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for FrameDelegate {
            #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
            unsafe fn did_output_frame(
                &self,
                _output: &AVCaptureOutput,
                sample_buffer: &CMSampleBuffer,
                _connection: &AVCaptureConnection,
            ) {
                self.process_frame(sample_buffer);
            }
        }
    );

    impl FrameDelegate {
        fn new(
            preview_layer: &Retained<AVCaptureVideoPreviewLayer>,
            overlay_layer: &Retained<CAShapeLayer>,
            face_id: Option<Arc<FaceIdClient>>,
        ) -> Retained<Self> {
            let ivars = FrameDelegateIvars {
                preview_layer: Retained::as_ptr(preview_layer) as usize,
                overlay_layer: Retained::as_ptr(overlay_layer) as usize,
                last_processed: Mutex::new(None),
                last_label_refresh: Mutex::new(None),
                tracker: Mutex::new(FaceTracker::default()),
                face_id,
                reported_error: AtomicBool::new(false),
            };
            let this = Self::alloc().set_ivars(ivars);
            unsafe { msg_send![super(this), init] }
        }

        fn process_frame(&self, sample_buffer: &CMSampleBuffer) {
            let mut last_processed = self
                .ivars()
                .last_processed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if last_processed.is_some_and(|last| last.elapsed() < FRAME_INTERVAL) {
                return;
            }
            let observed_at = Instant::now();
            *last_processed = Some(observed_at);
            drop(last_processed);

            let Some(pixel_buffer) = (unsafe { sample_buffer.image_buffer() }) else {
                return;
            };
            let detector = AppleVisionDetector;
            let detection = match detector.detect_pixel_buffer(&pixel_buffer) {
                Ok(detection) => detection,
                Err(error) => {
                    if !self.ivars().reported_error.swap(true, Ordering::Relaxed) {
                        eprintln!("Vision frame processing failed: {error}");
                    }
                    return;
                }
            };
            let inference_seconds = observed_at.elapsed().as_secs_f64();
            let observed_faces = match self.ivars().tracker.lock() {
                Ok(mut tracker) => tracker.update_at(detection.faces, observed_at),
                Err(_) => {
                    eprintln!("face tracker state is unavailable");
                    return;
                }
            };
            if let Some(face_id) = &self.ivars().face_id {
                submit_face_ids(&pixel_buffer, &observed_faces, face_id);
            }
            let tracked_faces = observed_faces
                .iter()
                .map(|face| face.predicted(inference_seconds + PRESENTATION_DELAY_SECONDS))
                .collect::<Vec<_>>();
            let face_id_matches = self
                .ivars()
                .face_id
                .as_ref()
                .map(|client| {
                    tracked_faces
                        .iter()
                        .filter_map(|face| {
                            client
                                .identity_for_track(face.track_id)
                                .map(|identity| (face.track_id, identity))
                        })
                        .collect::<HashMap<_, _>>()
                })
                .unwrap_or_default();
            let capture_status = self
                .ivars()
                .face_id
                .as_ref()
                .and_then(|client| client.capture_status());
            let refresh_labels = {
                let mut last_refresh = self
                    .ivars()
                    .last_label_refresh
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if tracked_faces.is_empty() {
                    *last_refresh = None;
                    true
                } else if last_refresh.is_none_or(|last| last.elapsed() >= LABEL_INTERVAL) {
                    *last_refresh = Some(Instant::now());
                    true
                } else {
                    false
                }
            };

            let preview_layer = self.ivars().preview_layer;
            let overlay_layer = self.ivars().overlay_layer;
            DispatchQueue::main().exec_async(move || {
                let preview = unsafe { &*(preview_layer as *const AVCaptureVideoPreviewLayer) };
                let overlay = unsafe { &*(overlay_layer as *const CAShapeLayer) };
                update_overlay(
                    preview,
                    overlay,
                    &tracked_faces,
                    &face_id_matches,
                    capture_status.as_ref(),
                    refresh_labels,
                    inference_seconds * 1_000.0,
                );
            });
        }
    }

    #[derive(Debug, Default)]
    struct AppDelegateIvars {
        window: OnceCell<Retained<NSWindow>>,
        session: OnceCell<Retained<AVCaptureSession>>,
        preview_layer: OnceCell<Retained<AVCaptureVideoPreviewLayer>>,
        overlay_layer: OnceCell<Retained<CAShapeLayer>>,
        frame_delegate: OnceCell<Retained<FrameDelegate>>,
        video_output: OnceCell<Retained<AVCaptureVideoDataOutput>>,
        face_id: Option<Arc<FaceIdClient>>,
    }

    define_class!(
        // SAFETY: NSObject has no subclassing requirements and the class has no Drop impl.
        #[unsafe(super = NSObject)]
        #[thread_kind = MainThreadOnly]
        #[ivars = AppDelegateIvars]
        struct AppDelegate;

        // SAFETY: NSObjectProtocol has no additional safety requirements.
        unsafe impl NSObjectProtocol for AppDelegate {}

        // SAFETY: The method signatures match NSApplicationDelegate.
        unsafe impl NSApplicationDelegate for AppDelegate {
            #[unsafe(method(applicationDidFinishLaunching:))]
            fn did_finish_launching(&self, notification: &NSNotification) {
                let app = notification
                    .object()
                    .and_then(|object| object.downcast::<NSApplication>().ok())
                    .expect("launch notification must contain NSApplication");

                if let Err(error) = self.create_camera_window(&app) {
                    eprintln!("error: {error}");
                    app.terminate(None);
                }
            }
        }

        // SAFETY: The method signatures match NSWindowDelegate.
        unsafe impl NSWindowDelegate for AppDelegate {
            #[unsafe(method(windowDidResize:))]
            fn window_did_resize(&self, _notification: &NSNotification) {
                self.resize_layers();
            }

            #[unsafe(method(windowWillClose:))]
            fn window_will_close(&self, _notification: &NSNotification) {
                NSApplication::sharedApplication(self.mtm()).terminate(None);
            }
        }
    );

    impl AppDelegate {
        fn new(mtm: MainThreadMarker, face_id: Option<Arc<FaceIdClient>>) -> Retained<Self> {
            let this = Self::alloc(mtm).set_ivars(AppDelegateIvars {
                face_id,
                ..AppDelegateIvars::default()
            });
            unsafe { msg_send![super(this), init] }
        }

        fn create_camera_window(&self, app: &NSApplication) -> Result<(), String> {
            let mtm = self.mtm();
            let rect = CGRect::new(CGPoint::ZERO, CGSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
            let window = unsafe {
                NSWindow::initWithContentRect_styleMask_backing_defer(
                    NSWindow::alloc(mtm),
                    rect,
                    NSWindowStyleMask::Titled
                        | NSWindowStyleMask::Closable
                        | NSWindowStyleMask::Miniaturizable
                        | NSWindowStyleMask::Resizable,
                    NSBackingStoreType::Buffered,
                    false,
                )
            };
            unsafe { window.setReleasedWhenClosed(false) };
            window.setTitle(ns_string!("FaceFeature — Local Vision"));
            window.setDelegate(Some(ProtocolObject::from_ref(self)));

            let content_view = window
                .contentView()
                .ok_or("window did not create a content view")?;
            content_view.setWantsLayer(true);
            let root_layer = content_view
                .layer()
                .ok_or("window content view did not create a Core Animation layer")?;

            let session: Retained<AVCaptureSession> =
                unsafe { msg_send![AVCaptureSession::alloc(), init] };
            let video_type = video_media_type()?;
            let device = unsafe { AVCaptureDevice::defaultDeviceWithMediaType(video_type) }
                .ok_or("no camera is available")?;
            let input = unsafe { AVCaptureDeviceInput::deviceInputWithDevice_error(&device) }
                .map_err(|error| error.to_string())?;
            let input_for_session: Retained<AVCaptureInput> = input.into_super();

            let output = unsafe { AVCaptureVideoDataOutput::new() };
            unsafe { output.setAlwaysDiscardsLateVideoFrames(true) };
            configure_bgra_output(&output);
            let output_for_session: Retained<AVCaptureOutput> = output.clone().into_super();

            unsafe {
                session.beginConfiguration();
                if session.canSetSessionPreset(AVCaptureSessionPreset1280x720) {
                    session.setSessionPreset(AVCaptureSessionPreset1280x720);
                }
                if !session.canAddInput(&input_for_session) {
                    session.commitConfiguration();
                    return Err("camera input cannot be added to the capture session".to_owned());
                }
                session.addInput(&input_for_session);
                if !session.canAddOutput(&output_for_session) {
                    session.commitConfiguration();
                    return Err("video output cannot be added to the capture session".to_owned());
                }
                session.addOutput(&output_for_session);
                session.commitConfiguration();
            }

            let preview = unsafe { AVCaptureVideoPreviewLayer::layerWithSession(&session) };
            if let Some(gravity) = unsafe { AVLayerVideoGravityResizeAspectFill } {
                unsafe { preview.setVideoGravity(gravity) };
            }
            disable_mirroring(unsafe { preview.connection() });
            disable_mirroring(unsafe { output.connectionWithMediaType(video_type) });

            let overlay = CAShapeLayer::layer();
            let green = CGColor::new_generic_rgb(0.15, 1.0, 0.42, 0.95);
            overlay.setStrokeColor(Some(&green));
            overlay.setFillColor(None);
            overlay.setLineWidth(2.0);
            unsafe {
                overlay.setLineCap(kCALineCapRound);
                overlay.setLineJoin(kCALineJoinRound);
            }

            preview.setFrame(content_view.bounds());
            overlay.setFrame(content_view.bounds());
            root_layer.addSublayer(&preview);
            root_layer.addSublayer(&overlay);

            let frame_delegate =
                FrameDelegate::new(&preview, &overlay, self.ivars().face_id.clone());
            let frame_queue =
                DispatchQueue::new("dev.facefeature.camera.frames", DispatchQueueAttr::SERIAL);
            unsafe {
                output.setSampleBufferDelegate_queue(
                    Some(ProtocolObject::from_ref(&*frame_delegate)),
                    Some(&frame_queue),
                );
            }

            self.ivars().window.set(window.clone()).unwrap();
            self.ivars().session.set(session.clone()).unwrap();
            self.ivars().preview_layer.set(preview).unwrap();
            self.ivars().overlay_layer.set(overlay).unwrap();
            self.ivars().frame_delegate.set(frame_delegate).unwrap();
            self.ivars().video_output.set(output).unwrap();

            window.center();
            window.makeKeyAndOrderFront(None);
            app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
            #[allow(deprecated)]
            app.activateIgnoringOtherApps(true);

            let session_pointer = Retained::as_ptr(&session) as usize;
            frame_queue.exec_async(move || {
                let session = unsafe { &*(session_pointer as *const AVCaptureSession) };
                unsafe { session.startRunning() };
            });

            Ok(())
        }

        fn resize_layers(&self) {
            let Some(window) = self.ivars().window.get() else {
                return;
            };
            let Some(view) = window.contentView() else {
                return;
            };
            let bounds = view.bounds();
            CATransaction::begin();
            CATransaction::setDisableActions(true);
            if let Some(preview) = self.ivars().preview_layer.get() {
                preview.setFrame(bounds);
            }
            if let Some(overlay) = self.ivars().overlay_layer.get() {
                overlay.setFrame(bounds);
            }
            CATransaction::commit();
        }
    }

    fn video_media_type() -> Result<&'static objc2_av_foundation::AVMediaType, String> {
        unsafe { AVMediaTypeVideo }.ok_or_else(|| "AVMediaTypeVideo is unavailable".to_owned())
    }

    fn ensure_camera_access() -> Result<(), String> {
        let video_type = video_media_type()?;
        let status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(video_type) };
        match status {
            AVAuthorizationStatus::Authorized => Ok(()),
            AVAuthorizationStatus::Denied => Err(
                "camera access is denied; enable it for your terminal in System Settings > Privacy & Security > Camera"
                    .to_owned(),
            ),
            AVAuthorizationStatus::Restricted => {
                Err("camera access is restricted by system policy".to_owned())
            }
            AVAuthorizationStatus::NotDetermined => {
                let (sender, receiver) = mpsc::sync_channel(1);
                let completion: RcBlock<dyn Fn(Bool)> = RcBlock::new(move |granted: Bool| {
                    let _ = sender.send(granted.as_bool());
                });
                unsafe {
                    AVCaptureDevice::requestAccessForMediaType_completionHandler(
                        video_type,
                        &completion,
                    );
                }
                match receiver.recv() {
                    Ok(true) => Ok(()),
                    Ok(false) => Err("camera permission was not granted".to_owned()),
                    Err(error) => Err(format!("camera permission request failed: {error}")),
                }
            }
            _ => Err("macOS returned an unknown camera authorization status".to_owned()),
        }
    }

    fn disable_mirroring(connection: Option<Retained<AVCaptureConnection>>) {
        let Some(connection) = connection else {
            return;
        };
        unsafe {
            connection.setAutomaticallyAdjustsVideoMirroring(false);
            if connection.isVideoMirroringSupported() {
                connection.setVideoMirrored(false);
            }
        }
    }

    fn configure_bgra_output(output: &AVCaptureVideoDataOutput) {
        let pixel_format = NSNumber::new_u32(kCVPixelFormatType_32BGRA);
        let key_cf = unsafe { kCVPixelBufferPixelFormatTypeKey };
        // CFString/NSString are toll-free bridged and NSDictionary's generic arguments are
        // erased by Objective-C. The runtime value is an NSNumber, which is an AnyObject.
        let key = unsafe { &*(key_cf as *const CFString as *const NSString) };
        let typed = NSDictionary::<NSString, NSNumber>::from_slices(&[key], &[&pixel_format]);
        let settings = unsafe {
            &*((&*typed as *const NSDictionary<NSString, NSNumber>)
                as *const NSDictionary<NSString, AnyObject>)
        };
        unsafe { output.setVideoSettings(Some(settings)) };
    }

    fn submit_face_ids(
        pixel_buffer: &CVPixelBuffer,
        tracked_faces: &[TrackedFace],
        client: &FaceIdClient,
    ) {
        if client.is_guided_capture() {
            submit_guided_capture(pixel_buffer, tracked_faces, client);
            return;
        }
        let observed_at = Instant::now();
        let candidates = tracked_faces
            .iter()
            .filter(|face| client.wants_track(face.track_id, &face.geometry, observed_at))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return;
        }

        let lock_flags = CVPixelBufferLockFlags::ReadOnly;
        if unsafe { CVPixelBufferLockBaseAddress(pixel_buffer, lock_flags) } != kCVReturnSuccess {
            return;
        }
        let width = CVPixelBufferGetWidth(pixel_buffer);
        let height = CVPixelBufferGetHeight(pixel_buffer);
        let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
        let base_address = CVPixelBufferGetBaseAddress(pixel_buffer).cast::<u8>();
        if !base_address.is_null() {
            let pixels =
                unsafe { std::slice::from_raw_parts(base_address, bytes_per_row * height) };
            for tracked_face in candidates {
                if let Ok(tensor) = aligned_sface_tensor_bgra(
                    pixels,
                    width,
                    height,
                    bytes_per_row,
                    &tracked_face.geometry,
                ) {
                    client.try_submit(
                        tracked_face.track_id,
                        tensor,
                        &tracked_face.geometry,
                        observed_at,
                    );
                }
            }
        }
        let _ = unsafe { CVPixelBufferUnlockBaseAddress(pixel_buffer, lock_flags) };
    }

    fn submit_guided_capture(
        pixel_buffer: &CVPixelBuffer,
        tracked_faces: &[TrackedFace],
        client: &FaceIdClient,
    ) {
        let tracked_face = match tracked_faces {
            [] => {
                client.report_capture_problem("NO FACE — move one person into view");
                return;
            }
            [tracked_face] => tracked_face,
            _ => {
                client.report_capture_problem("MULTIPLE FACES — only one person is allowed");
                return;
            }
        };
        let Some(request) = client.capture_request(&tracked_face.geometry, Instant::now()) else {
            return;
        };
        let lock_flags = CVPixelBufferLockFlags::ReadOnly;
        if unsafe { CVPixelBufferLockBaseAddress(pixel_buffer, lock_flags) } != kCVReturnSuccess {
            client.release_capture("could not lock camera pixels");
            return;
        }
        let width = CVPixelBufferGetWidth(pixel_buffer);
        let height = CVPixelBufferGetHeight(pixel_buffer);
        let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
        let base_address = CVPixelBufferGetBaseAddress(pixel_buffer).cast::<u8>();
        if base_address.is_null() {
            client.release_capture("camera frame has no pixel data");
        } else {
            let pixels =
                unsafe { std::slice::from_raw_parts(base_address, bytes_per_row * height) };
            match aligned_sface_tensor_bgra(
                pixels,
                width,
                height,
                bytes_per_row,
                &tracked_face.geometry,
            ) {
                Ok(tensor) => client.try_submit_capture(tracked_face.track_id, tensor, request),
                Err(error) => client.release_capture(&format!("face alignment failed: {error}")),
            }
        }
        let _ = unsafe { CVPixelBufferUnlockBaseAddress(pixel_buffer, lock_flags) };
    }

    fn update_overlay(
        preview: &AVCaptureVideoPreviewLayer,
        overlay: &CAShapeLayer,
        tracked_faces: &[TrackedFace],
        face_id_matches: &HashMap<u64, FaceIdentityMatch>,
        capture_status: Option<&FaceCaptureStatus>,
        refresh_labels: bool,
        inference_milliseconds: f64,
    ) {
        let path = CGMutablePath::new();
        for tracked_face in tracked_faces {
            let face = &tracked_face.geometry;
            add_face_box(&path, preview, face.bounding_box);
            for region in &face.landmarks {
                if region.kind == LandmarkKind::AllPoints || region.points.is_empty() {
                    continue;
                }
                if matches!(
                    region.kind,
                    LandmarkKind::LeftPupil | LandmarkKind::RightPupil
                ) {
                    add_point_marker(&path, preview, region.points[0]);
                } else {
                    add_polyline(&path, preview, region.kind, &region.points);
                }
            }
        }

        CATransaction::begin();
        CATransaction::setDisableActions(true);
        overlay.setPath(Some(&path));
        if refresh_labels {
            replace_face_labels(
                preview,
                overlay,
                tracked_faces,
                face_id_matches,
                capture_status,
                inference_milliseconds,
            );
        }
        CATransaction::commit();
    }

    fn replace_face_labels(
        preview: &AVCaptureVideoPreviewLayer,
        overlay: &CAShapeLayer,
        tracked_faces: &[TrackedFace],
        face_id_matches: &HashMap<u64, FaceIdentityMatch>,
        capture_status: Option<&FaceCaptureStatus>,
        inference_milliseconds: f64,
    ) {
        clear_face_labels(overlay);
        if let Some(status) = capture_status {
            add_capture_banner(preview, overlay, status);
        }

        for tracked_face in tracked_faces {
            let face = &tracked_face.geometry;
            let bounds = face.bounding_box;
            let center_x = bounds.x + bounds.width / 2.0;
            let center_y = bounds.y + bounds.height / 2.0;
            let point_count = face
                .landmarks
                .iter()
                .find(|region| region.kind == LandmarkKind::AllPoints)
                .map_or(0, |region| region.points.len());
            let pose = format_pose(
                face.yaw_radians,
                face.pitch_radians,
                landmark_pitch_degrees(face),
                face.roll_radians,
            );
            let identity = face_id_matches
                .get(&tracked_face.track_id)
                .map(format_identity_label)
                .unwrap_or_default();
            let value = NSString::from_str(&format!(
                "FACE {}  {:.0}%  {}pts  {:.0}ms{}\nbox x {:.3}  y {:.3}\nsize w {:.3}  h {:.3}\ncenter {:.3}, {:.3}\n{}",
                tracked_face.track_id,
                face.confidence * 100.0,
                point_count,
                inference_milliseconds,
                identity,
                bounds.x,
                bounds.y,
                bounds.width,
                bounds.height,
                center_x,
                center_y,
                pose,
            ));

            let label = CATextLayer::layer();
            unsafe { label.setString(Some(&value)) };
            label.setFontSize(12.0);
            label.setWrapped(true);
            label.setContentsScale(2.0);
            let foreground = CGColor::new_generic_rgb(0.82, 1.0, 0.88, 1.0);
            let background = CGColor::new_generic_rgb(0.01, 0.07, 0.035, 0.84);
            label.setForegroundColor(Some(&foreground));
            label.setBackgroundColor(Some(&background));
            label.setCornerRadius(5.0);
            label.setFrame(face_label_frame(preview, bounds));
            overlay.addSublayer(&label);
        }
    }

    fn add_capture_banner(
        preview: &AVCaptureVideoPreviewLayer,
        overlay: &CAShapeLayer,
        status: &FaceCaptureStatus,
    ) {
        let yaw = status
            .yaw_degrees
            .map(|value| format!("{value:.1}°"))
            .unwrap_or_else(|| "--".to_owned());
        let pitch = status
            .pitch_degrees
            .map(|value| {
                if status.pitch_estimated {
                    format!("~{value:.1}°")
                } else {
                    format!("{value:.1}°")
                }
            })
            .unwrap_or_else(|| "--".to_owned());
        let value = NSString::from_str(&format!(
            "ENROLL {}   pose {}/{}   sample {}/{}\n{}\nyaw {}  pitch {}  quality {:.0}%   {}",
            status.subject,
            status.pose_number,
            status.pose_count,
            status.samples,
            status.samples_required,
            status.instruction,
            yaw,
            pitch,
            status.quality * 100.0,
            status.message,
        ));
        let banner = CATextLayer::layer();
        unsafe { banner.setString(Some(&value)) };
        banner.setFontSize(16.0);
        banner.setAlignmentMode(unsafe { objc2_quartz_core::kCAAlignmentCenter });
        banner.setWrapped(true);
        banner.setContentsScale(2.0);
        let foreground = if status.alert {
            CGColor::new_generic_rgb(1.0, 0.94, 0.94, 1.0)
        } else if status.completed {
            CGColor::new_generic_rgb(0.7, 1.0, 0.75, 1.0)
        } else {
            CGColor::new_generic_rgb(1.0, 0.95, 0.72, 1.0)
        };
        let background = if status.alert {
            CGColor::new_generic_rgb(0.72, 0.03, 0.04, 0.94)
        } else {
            CGColor::new_generic_rgb(0.02, 0.03, 0.04, 0.9)
        };
        banner.setForegroundColor(Some(&foreground));
        banner.setBackgroundColor(Some(&background));
        banner.setCornerRadius(8.0);
        let width = 540.0_f64.min(preview.bounds().size.width - 24.0);
        let height = preview.bounds().size.height;
        banner.setFrame(CGRect::new(
            CGPoint::new(
                (preview.bounds().size.width - width) / 2.0,
                (height - 78.0 - 18.0).max(0.0),
            ),
            CGSize::new(width, 78.0),
        ));
        overlay.addSublayer(&banner);
    }

    fn format_identity_label(identity: &FaceIdentityMatch) -> String {
        let name = identity
            .display_name()
            .replace(['\n', '\r'], " ")
            .chars()
            .take(32)
            .collect::<String>();
        if !identity.persisted {
            format!(
                "\nID Unknown  unmatched  best {:.3}\nfp {}",
                identity.similarity, identity.fingerprint
            )
        } else if identity.is_new {
            format!(
                "\nID {}  NEW  best {:.3}\nfp {}",
                name, identity.similarity, identity.fingerprint
            )
        } else {
            format!(
                "\nID {}  similarity {:.3}\nfp {}",
                name, identity.similarity, identity.fingerprint
            )
        }
    }

    fn clear_face_labels(overlay: &CAShapeLayer) {
        // The overlay owns only transient text sublayers. Clear them atomically instead of
        // removing items from Core Animation's live `sublayers` array while iterating it.
        unsafe { overlay.setSublayers(None) };
    }

    fn format_pose(
        yaw: Option<f64>,
        pitch: Option<f64>,
        estimated_pitch_degrees: Option<f64>,
        roll: Option<f64>,
    ) -> String {
        fn degrees(value: Option<f64>) -> String {
            value
                .map(|radians| format!("{:.1}\u{00b0}", radians.to_degrees()))
                .unwrap_or_else(|| "--".to_owned())
        }

        let pitch = pitch
            .map(|radians| format!("{:.1}\u{00b0}", radians.to_degrees()))
            .or_else(|| estimated_pitch_degrees.map(|degrees| format!("~{degrees:.1}\u{00b0}")))
            .unwrap_or_else(|| "--".to_owned());

        format!(
            "yaw {}  pitch {}  roll {}",
            degrees(yaw),
            pitch,
            degrees(roll)
        )
    }

    fn face_label_frame(
        preview: &AVCaptureVideoPreviewLayer,
        bounds: facefeature::BoundingBox,
    ) -> CGRect {
        const LABEL_WIDTH: f64 = 286.0;
        const LABEL_HEIGHT: f64 = 112.0;
        const GAP: f64 = 8.0;

        let corners = [
            layer_point(
                preview,
                Point {
                    x: bounds.x,
                    y: bounds.y,
                },
            ),
            layer_point(
                preview,
                Point {
                    x: bounds.x + bounds.width,
                    y: bounds.y + bounds.height,
                },
            ),
        ];
        let min_x = corners[0].x.min(corners[1].x);
        let max_x = corners[0].x.max(corners[1].x);
        let max_y = corners[0].y.max(corners[1].y);
        let preview_width = preview.bounds().size.width;
        let preview_height = preview.bounds().size.height;

        let mut x = max_x + GAP;
        if x + LABEL_WIDTH > preview_width {
            x = min_x - LABEL_WIDTH - GAP;
        }
        if x < 0.0 {
            x = (min_x + GAP).min((preview_width - LABEL_WIDTH).max(0.0));
        }
        let y = (max_y - LABEL_HEIGHT)
            .max(0.0)
            .min((preview_height - LABEL_HEIGHT).max(0.0));

        CGRect::new(CGPoint::new(x, y), CGSize::new(LABEL_WIDTH, LABEL_HEIGHT))
    }

    fn add_face_box(
        path: &CGMutablePath,
        preview: &AVCaptureVideoPreviewLayer,
        bounds: facefeature::BoundingBox,
    ) {
        let corners = [
            Point {
                x: bounds.x,
                y: bounds.y,
            },
            Point {
                x: bounds.x + bounds.width,
                y: bounds.y,
            },
            Point {
                x: bounds.x + bounds.width,
                y: bounds.y + bounds.height,
            },
            Point {
                x: bounds.x,
                y: bounds.y + bounds.height,
            },
        ];
        add_polyline(path, preview, LandmarkKind::FaceContour, &corners);
        CGMutablePath::close_subpath(Some(path));
    }

    fn add_polyline(
        path: &CGMutablePath,
        preview: &AVCaptureVideoPreviewLayer,
        kind: LandmarkKind,
        points: &[Point],
    ) {
        let Some(first) = points.first() else {
            return;
        };
        let first = layer_point(preview, *first);
        unsafe { CGMutablePath::move_to_point(Some(path), std::ptr::null(), first.x, first.y) };
        for point in &points[1..] {
            let point = layer_point(preview, *point);
            unsafe {
                CGMutablePath::add_line_to_point(Some(path), std::ptr::null(), point.x, point.y)
            };
        }
        if matches!(
            kind,
            LandmarkKind::LeftEye
                | LandmarkKind::RightEye
                | LandmarkKind::Nose
                | LandmarkKind::OuterLips
                | LandmarkKind::InnerLips
        ) {
            CGMutablePath::close_subpath(Some(path));
        }
    }

    fn add_point_marker(path: &CGMutablePath, preview: &AVCaptureVideoPreviewLayer, point: Point) {
        let point = layer_point(preview, point);
        let marker = CGRect::new(
            CGPoint::new(point.x - 2.5, point.y - 2.5),
            CGSize::new(5.0, 5.0),
        );
        unsafe { CGMutablePath::add_ellipse_in_rect(Some(path), std::ptr::null(), marker) };
    }

    fn layer_point(preview: &AVCaptureVideoPreviewLayer, point: Point) -> CGPoint {
        let capture_point = vision_to_capture_point(point);
        unsafe { preview.pointForCaptureDevicePointOfInterest(capture_point) }
    }

    fn vision_to_capture_point(point: Point) -> CGPoint {
        // On macOS, both Vision and the preview layer conversion accept normalized image-space
        // points here. AVCaptureVideoPreviewLayer then handles the layer origin and aspect-fill.
        CGPoint::new(point.x, point.y)
    }

    #[derive(Debug, Default, PartialEq)]
    struct CameraOptions {
        face_id_model: Option<PathBuf>,
        face_id_database: Option<PathBuf>,
        face_id_mode: Option<FaceIdDatabaseMode>,
        capture_requested: bool,
        capture_target: Option<FaceCaptureTarget>,
        benchmark_iterations: Option<usize>,
        show_help: bool,
    }

    fn parse_options(arguments: impl IntoIterator<Item = String>) -> Result<CameraOptions, String> {
        let mut options = CameraOptions::default();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--face-id" => {
                    options.face_id_model =
                        Some(PathBuf::from("models/face_recognition_sface_2021dec.onnx"));
                    options
                        .face_id_database
                        .get_or_insert_with(|| PathBuf::from("data/face_identities.sqlite3"));
                }
                "--face-id-model" => {
                    let path = arguments
                        .next()
                        .ok_or("--face-id-model requires a model path")?;
                    options.face_id_model = Some(PathBuf::from(path));
                    options
                        .face_id_database
                        .get_or_insert_with(|| PathBuf::from("data/face_identities.sqlite3"));
                }
                "--face-id-db" => {
                    let path = arguments
                        .next()
                        .ok_or("--face-id-db requires a database path")?;
                    options.face_id_database = Some(PathBuf::from(path));
                    options.face_id_model.get_or_insert_with(|| {
                        PathBuf::from("models/face_recognition_sface_2021dec.onnx")
                    });
                }
                "--read-only" => {
                    if options.face_id_mode == Some(FaceIdDatabaseMode::ReadWrite) {
                        return Err("--read-only and --capture cannot be used together".to_owned());
                    }
                    options.face_id_mode = Some(FaceIdDatabaseMode::ReadOnly);
                    enable_face_id_defaults(&mut options);
                }
                "--capture" => {
                    if options.face_id_mode == Some(FaceIdDatabaseMode::ReadOnly) {
                        return Err("--read-only and --capture cannot be used together".to_owned());
                    }
                    options.face_id_mode = Some(FaceIdDatabaseMode::ReadWrite);
                    options.capture_requested = true;
                    enable_face_id_defaults(&mut options);
                }
                "--benchmark" => {
                    options.benchmark_iterations.get_or_insert(100);
                    options.face_id_model.get_or_insert_with(|| {
                        PathBuf::from("models/face_recognition_sface_2021dec.onnx")
                    });
                }
                "--benchmark-iterations" => {
                    let value = arguments
                        .next()
                        .ok_or("--benchmark-iterations requires a number")?;
                    let iterations = value
                        .parse::<usize>()
                        .ok()
                        .filter(|value| (1..=100_000).contains(value))
                        .ok_or("--benchmark-iterations must be between 1 and 100000")?;
                    options.benchmark_iterations = Some(iterations);
                    options.face_id_model.get_or_insert_with(|| {
                        PathBuf::from("models/face_recognition_sface_2021dec.onnx")
                    });
                }
                "--name" => {
                    let name = arguments.next().ok_or("--name requires a value")?;
                    if name.trim().is_empty() {
                        return Err("--name cannot be empty".to_owned());
                    }
                    if options.capture_target.is_some() {
                        return Err("use either --name or --person, not both".to_owned());
                    }
                    options.capture_target = Some(FaceCaptureTarget::New { name });
                }
                "--person" => {
                    let value = arguments.next().ok_or("--person requires an ID")?;
                    let person_id = value
                        .parse::<u64>()
                        .ok()
                        .filter(|id| *id > 0)
                        .ok_or("--person must be a positive integer")?;
                    if options.capture_target.is_some() {
                        return Err("use either --name or --person, not both".to_owned());
                    }
                    options.capture_target = Some(FaceCaptureTarget::Existing { person_id });
                }
                "--help" | "-h" => options.show_help = true,
                unknown => return Err(format!("unknown camera option: {unknown}")),
            }
        }
        if options.capture_requested && options.capture_target.is_none() {
            return Err("--capture requires --name NAME or --person ID".to_owned());
        }
        if !options.capture_requested && options.capture_target.is_some() {
            return Err("--name and --person can only be used with --capture".to_owned());
        }
        if options.capture_requested && options.benchmark_iterations.is_some() {
            return Err("--benchmark and --capture cannot be used together".to_owned());
        }
        Ok(options)
    }

    fn enable_face_id_defaults(options: &mut CameraOptions) {
        options
            .face_id_model
            .get_or_insert_with(|| PathBuf::from("models/face_recognition_sface_2021dec.onnx"));
        options
            .face_id_database
            .get_or_insert_with(|| PathBuf::from("data/face_identities.sqlite3"));
    }

    fn print_help() {
        println!(
            "facefeature-camera\n\nUSAGE:\n    facefeature-camera --face-id [OPTIONS]\n    facefeature-camera --read-only [OPTIONS]\n    facefeature-camera --capture (--name NAME | --person ID) [OPTIONS]\n    facefeature-camera --benchmark [OPTIONS]\n\nOPTIONS:\n    --face-id                 Automatic matching and enrollment\n    --read-only               Match from SQLite without inserting or updating rows\n    --capture                 Guided center/left/right/up/down enrollment\n    --name NAME               Create a new named identity during guided capture\n    --person ID               Replace/add guided samples for an existing identity\n    --benchmark               Benchmark headless SFace/Core ML inference\n    --benchmark-iterations N  Timed iterations (default: 100)\n    --face-id-model PATH      Use a custom SFace ONNX model\n    --face-id-db PATH         Use a custom SQLite identity gallery path\n    -h, --help                Print this help"
        );
    }

    fn system_value(program: &str, arguments: &[&str]) -> String {
        Command::new(program)
            .args(arguments)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "unknown".to_owned())
    }

    fn hardware_summary() -> (String, String) {
        let output = Command::new("/usr/sbin/system_profiler")
            .args(["SPHardwareDataType", "-detailLevel", "mini"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .unwrap_or_default();
        let field = |name: &str| {
            output.lines().find_map(|line| {
                line.trim()
                    .strip_prefix(name)
                    .map(|value| value.trim().to_owned())
            })
        };
        let architecture = system_value("/usr/bin/uname", &["-m"]);
        let chip = field("Chip:").unwrap_or_else(|| architecture.clone());
        let model = field("Model Name:").unwrap_or_else(|| "Mac".to_owned());
        let memory = field("Memory:").unwrap_or_else(|| "unknown".to_owned());
        (format!("{model} | {chip} | {architecture}"), memory)
    }

    fn run_benchmark(model_path: &std::path::Path, iterations: usize) -> Result<(), String> {
        let (hardware, memory) = hardware_summary();
        let macos = system_value("/usr/bin/sw_vers", &["-productVersion"]);
        let cache_dir = PathBuf::from("target/face-id-coreml-cache");
        println!("SFace/Core ML benchmark");
        println!("hardware: {hardware} | memory: {memory} | macOS: {macos}");
        println!("model: {}", model_path.display());
        println!("backend: CoreML | compute units: all | input: 1x3x112x112");
        let report = benchmark_sface_coreml(model_path, &cache_dir, iterations)?;
        println!(
            "session initialization: {:.3} ms",
            report.session_initialization_ms
        );
        println!("first inference:       {:.3} ms", report.first_inference_ms);
        println!(
            "steady state ({} runs after {} warmups):",
            report.iterations, report.warmup_iterations
        );
        println!(
            "  average:             {:.3} ms",
            report.average_inference_ms
        );
        println!(
            "  median (p50):        {:.3} ms",
            report.median_inference_ms
        );
        println!("  p95:                 {:.3} ms", report.p95_inference_ms);
        println!(
            "  min / max:           {:.3} / {:.3} ms",
            report.minimum_inference_ms, report.maximum_inference_ms
        );
        println!(
            "  throughput:          {:.1} embeddings/s",
            report.embeddings_per_second
        );
        println!("embedding dimensions:  {}", report.embedding_dimensions);
        println!("note: synthetic input; this measures embedding compute, not camera or Vision");
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        let options = parse_options(env::args().skip(1))?;
        if options.show_help {
            print_help();
            return Ok(());
        }
        if let Some(iterations) = options.benchmark_iterations {
            let model_path = options
                .face_id_model
                .as_deref()
                .ok_or("benchmark model path is unavailable")?;
            return run_benchmark(model_path, iterations);
        }
        let face_id = options
            .face_id_model
            .as_deref()
            .zip(options.face_id_database.as_deref())
            .map(|(model_path, database_path)| {
                FaceIdClient::start(
                    model_path,
                    &PathBuf::from("target/face-id-coreml-cache"),
                    database_path,
                    options.face_id_mode.unwrap_or_default(),
                    options.capture_target.clone(),
                )
            })
            .transpose()?;
        if let (Some(model_path), Some(database_path)) =
            (options.face_id_model, options.face_id_database)
        {
            println!(
                "face-id enabled model={} database={} mode={} operation={} backend=CoreML compute=all threshold={:.3}",
                model_path.display(),
                database_path.display(),
                options.face_id_mode.unwrap_or_default().label(),
                if options.capture_requested {
                    "guided-capture"
                } else {
                    "recognition"
                },
                facefeature::face_id::SFACE_COSINE_THRESHOLD
            );
        }
        ensure_camera_access()?;
        let mtm = MainThreadMarker::new().ok_or("camera UI must start on the main thread")?;
        let app = NSApplication::sharedApplication(mtm);
        let delegate = AppDelegate::new(mtm, face_id);
        app.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
        app.run();
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn vision_points_keep_their_normalized_camera_position() {
            let mapped = vision_to_capture_point(Point { x: 0.2, y: 0.7 });
            assert!((mapped.x - 0.2).abs() < f64::EPSILON);
            assert!((mapped.y - 0.7).abs() < f64::EPSILON);
        }

        #[test]
        fn pose_label_uses_degrees_and_marks_missing_values() {
            assert_eq!(
                format_pose(
                    Some(std::f64::consts::FRAC_PI_2),
                    None,
                    Some(-4.25),
                    Some(-std::f64::consts::PI)
                ),
                "yaw 90.0\u{00b0}  pitch ~-4.2\u{00b0}  roll -180.0\u{00b0}"
            );
            assert_eq!(
                format_pose(None, Some(std::f64::consts::FRAC_PI_6), Some(-20.0), None,),
                "yaw --  pitch 30.0\u{00b0}  roll --"
            );
        }

        #[test]
        fn multiple_face_labels_are_cleared_atomically() {
            let overlay = CAShapeLayer::layer();
            for _ in 0..4 {
                overlay.addSublayer(&CATextLayer::layer());
            }
            assert_eq!(unsafe { overlay.sublayers() }.unwrap().count(), 4);

            clear_face_labels(&overlay);

            assert_eq!(
                unsafe { overlay.sublayers() }.map_or(0, |layers| layers.count()),
                0
            );
        }

        #[test]
        fn face_id_flag_uses_the_bundled_model() {
            let options = parse_options(["--face-id".to_owned()]).unwrap();
            assert_eq!(
                options.face_id_model,
                Some(PathBuf::from("models/face_recognition_sface_2021dec.onnx"))
            );
            assert_eq!(
                options.face_id_database,
                Some(PathBuf::from("data/face_identities.sqlite3"))
            );
        }

        #[test]
        fn face_id_model_flag_implies_face_id() {
            let options =
                parse_options(["--face-id-model".to_owned(), "/tmp/custom.onnx".to_owned()])
                    .unwrap();
            assert_eq!(
                options.face_id_model,
                Some(PathBuf::from("/tmp/custom.onnx"))
            );
            assert_eq!(
                options.face_id_database,
                Some(PathBuf::from("data/face_identities.sqlite3"))
            );
        }

        #[test]
        fn face_id_database_flag_implies_face_id() {
            let options =
                parse_options(["--face-id-db".to_owned(), "/tmp/people.sqlite3".to_owned()])
                    .unwrap();
            assert_eq!(
                options.face_id_model,
                Some(PathBuf::from("models/face_recognition_sface_2021dec.onnx"))
            );
            assert_eq!(
                options.face_id_database,
                Some(PathBuf::from("/tmp/people.sqlite3"))
            );
        }

        #[test]
        fn read_only_implies_face_id_and_selects_read_only_database() {
            let options = parse_options(["--read-only".to_owned()]).unwrap();
            assert!(options.face_id_model.is_some());
            assert!(options.face_id_database.is_some());
            assert_eq!(options.face_id_mode, Some(FaceIdDatabaseMode::ReadOnly));
        }

        #[test]
        fn capture_explicitly_selects_read_write_database() {
            let options = parse_options([
                "--capture".to_owned(),
                "--name".to_owned(),
                "Radit".to_owned(),
            ])
            .unwrap();
            assert!(options.face_id_model.is_some());
            assert!(options.face_id_database.is_some());
            assert_eq!(options.face_id_mode, Some(FaceIdDatabaseMode::ReadWrite));
            assert_eq!(
                options.capture_target,
                Some(FaceCaptureTarget::New {
                    name: "Radit".to_owned()
                })
            );
        }

        #[test]
        fn benchmark_is_headless_and_uses_the_bundled_model() {
            let options = parse_options(["--benchmark".to_owned()]).unwrap();
            assert_eq!(options.benchmark_iterations, Some(100));
            assert_eq!(
                options.face_id_model,
                Some(PathBuf::from("models/face_recognition_sface_2021dec.onnx"))
            );
            assert!(options.face_id_database.is_none());
        }

        #[test]
        fn benchmark_iterations_imply_benchmark_mode() {
            let options =
                parse_options(["--benchmark-iterations".to_owned(), "17".to_owned()]).unwrap();
            assert_eq!(options.benchmark_iterations, Some(17));
            assert!(options.face_id_model.is_some());
        }

        #[test]
        fn benchmark_and_capture_conflict() {
            let error = parse_options([
                "--benchmark".to_owned(),
                "--capture".to_owned(),
                "--name".to_owned(),
                "Test".to_owned(),
            ])
            .unwrap_err();
            assert!(error.contains("cannot be used together"));
        }

        #[test]
        fn capture_requires_a_name_or_existing_person() {
            let error = parse_options(["--capture".to_owned()]).unwrap_err();
            assert!(error.contains("requires --name NAME or --person ID"));
        }

        #[test]
        fn capture_can_update_an_existing_person() {
            let options = parse_options([
                "--capture".to_owned(),
                "--person".to_owned(),
                "7".to_owned(),
            ])
            .unwrap();
            assert_eq!(
                options.capture_target,
                Some(FaceCaptureTarget::Existing { person_id: 7 })
            );
        }

        #[test]
        fn read_only_and_capture_conflict() {
            let error =
                parse_options(["--capture".to_owned(), "--read-only".to_owned()]).unwrap_err();
            assert!(error.contains("cannot be used together"));
        }

        #[test]
        fn identity_label_contains_name_similarity_and_fingerprint() {
            let identity = FaceIdentityMatch {
                person_id: 7,
                name: Some("Radit".to_owned()),
                similarity: 0.812,
                fingerprint: "034a7a9d04edc8ce".to_owned(),
                is_new: false,
                persisted: true,
            };
            assert_eq!(
                format_identity_label(&identity),
                "\nID Radit  similarity 0.812\nfp 034a7a9d04edc8ce"
            );
        }

        #[test]
        fn unknown_read_only_identity_is_labeled_as_unmatched() {
            let identity = FaceIdentityMatch {
                person_id: 0,
                name: None,
                similarity: 0.271,
                fingerprint: "candidate1234567".to_owned(),
                is_new: true,
                persisted: false,
            };
            assert_eq!(
                format_identity_label(&identity),
                "\nID Unknown  unmatched  best 0.271\nfp candidate1234567"
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = macos::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("error: the native camera window currently requires macOS");
    std::process::exit(1);
}
