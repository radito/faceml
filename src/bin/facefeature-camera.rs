#[cfg(target_os = "macos")]
mod macos {
    use std::cell::OnceCell;
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
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
    use objc2_core_foundation::{CFRetained, CFString, CGPoint, CGRect, CGSize};
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
    const DEPTH_LAYER_COUNT: usize = 8;

    #[derive(Debug)]
    struct FrameDelegateIvars {
        preview_layer: usize,
        face_mask_layers: Vec<usize>,
        face_mask_mode: FaceMaskMode,
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
            face_mask_layers: &[Retained<CAShapeLayer>],
            face_mask_mode: FaceMaskMode,
            overlay_layer: &Retained<CAShapeLayer>,
            face_id: Option<Arc<FaceIdClient>>,
        ) -> Retained<Self> {
            let ivars = FrameDelegateIvars {
                preview_layer: Retained::as_ptr(preview_layer) as usize,
                face_mask_layers: face_mask_layers
                    .iter()
                    .map(|layer| Retained::as_ptr(layer) as usize)
                    .collect(),
                face_mask_mode,
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
            let face_mask_layers = self.ivars().face_mask_layers.clone();
            let face_mask_mode = self.ivars().face_mask_mode;
            let overlay_layer = self.ivars().overlay_layer;
            DispatchQueue::main().exec_async(move || {
                let preview = unsafe { &*(preview_layer as *const AVCaptureVideoPreviewLayer) };
                let face_mask_layers = face_mask_layers
                    .iter()
                    .map(|layer| unsafe { &*(*layer as *const CAShapeLayer) })
                    .collect::<Vec<_>>();
                let overlay = unsafe { &*(overlay_layer as *const CAShapeLayer) };
                update_overlay(
                    preview,
                    &face_mask_layers,
                    face_mask_mode,
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
        face_mask_layers: OnceCell<Vec<Retained<CAShapeLayer>>>,
        overlay_layer: OnceCell<Retained<CAShapeLayer>>,
        frame_delegate: OnceCell<Retained<FrameDelegate>>,
        video_output: OnceCell<Retained<AVCaptureVideoDataOutput>>,
        face_id: Option<Arc<FaceIdClient>>,
        face_mask: FaceMaskMode,
        mask_only: bool,
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
        fn new(
            mtm: MainThreadMarker,
            face_id: Option<Arc<FaceIdClient>>,
            face_mask: FaceMaskMode,
            mask_only: bool,
        ) -> Retained<Self> {
            let this = Self::alloc(mtm).set_ivars(AppDelegateIvars {
                face_id,
                face_mask,
                mask_only,
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
            if self.ivars().mask_only {
                let background = CGColor::new_generic_rgb(0.005, 0.012, 0.009, 1.0);
                root_layer.setBackgroundColor(Some(&background));
                preview.setHidden(true);
            }
            disable_mirroring(unsafe { preview.connection() });
            disable_mirroring(unsafe { output.connectionWithMediaType(video_type) });

            let overlay = CAShapeLayer::layer();
            let (overlay_alpha, overlay_width) = if self.ivars().face_mask == FaceMaskMode::Depth {
                (0.70, 1.15)
            } else {
                (0.95, 2.0)
            };
            let green = CGColor::new_generic_rgb(0.15, 1.0, 0.42, overlay_alpha);
            overlay.setStrokeColor(Some(&green));
            overlay.setFillColor(None);
            overlay.setLineWidth(overlay_width);
            unsafe {
                overlay.setLineCap(kCALineCapRound);
                overlay.setLineJoin(kCALineJoinRound);
            }

            let make_mask_layer = |alpha: f64, width: f64| {
                let layer = CAShapeLayer::layer();
                let stroke = CGColor::new_generic_rgb(0.15, 1.0, 0.42, alpha);
                layer.setFillColor(None);
                layer.setStrokeColor(Some(&stroke));
                layer.setLineWidth(width);
                unsafe {
                    layer.setLineCap(kCALineCapRound);
                    layer.setLineJoin(kCALineJoinRound);
                }
                layer
            };
            let face_mask_layers = match self.ivars().face_mask {
                FaceMaskMode::None => Vec::new(),
                FaceMaskMode::Polygon => vec![make_mask_layer(0.72, 0.85)],
                FaceMaskMode::Depth => (0..DEPTH_LAYER_COUNT)
                    .map(|index| {
                        let intensity = index as f64 / (DEPTH_LAYER_COUNT - 1) as f64;
                        make_mask_layer(0.22 + intensity * 0.56, 0.58 + intensity * 0.26)
                    })
                    .collect(),
            };

            preview.setFrame(content_view.bounds());
            for face_mask_layer in &face_mask_layers {
                face_mask_layer.setFrame(content_view.bounds());
            }
            overlay.setFrame(content_view.bounds());
            root_layer.addSublayer(&preview);
            for face_mask_layer in &face_mask_layers {
                root_layer.addSublayer(face_mask_layer);
            }
            root_layer.addSublayer(&overlay);

            let frame_delegate = FrameDelegate::new(
                &preview,
                &face_mask_layers,
                self.ivars().face_mask,
                &overlay,
                self.ivars().face_id.clone(),
            );
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
            self.ivars().face_mask_layers.set(face_mask_layers).unwrap();
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
            if let Some(face_mask_layers) = self.ivars().face_mask_layers.get() {
                for face_mask_layer in face_mask_layers {
                    face_mask_layer.setFrame(bounds);
                }
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
        face_mask_layers: &[&CAShapeLayer],
        face_mask_mode: FaceMaskMode,
        overlay: &CAShapeLayer,
        tracked_faces: &[TrackedFace],
        face_id_matches: &HashMap<u64, FaceIdentityMatch>,
        capture_status: Option<&FaceCaptureStatus>,
        refresh_labels: bool,
        inference_milliseconds: f64,
    ) {
        let face_mask_paths = face_mask_layers
            .iter()
            .map(|_| CGMutablePath::new())
            .collect::<Vec<_>>();
        for tracked_face in tracked_faces {
            match face_mask_mode {
                FaceMaskMode::None => {}
                FaceMaskMode::Polygon => {
                    if let Some(path) = face_mask_paths.first() {
                        add_polygon_face_mask(path, preview, &tracked_face.geometry);
                    }
                }
                FaceMaskMode::Depth => {
                    add_depth_face_mask(&face_mask_paths, preview, &tracked_face.geometry);
                }
            }
        }
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
        for (layer, path) in face_mask_layers.iter().zip(&face_mask_paths) {
            layer.setPath(Some(&path));
        }
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

    fn add_polygon_face_mask(
        path: &CGMutablePath,
        preview: &AVCaptureVideoPreviewLayer,
        face: &facefeature::FaceGeometry,
    ) {
        let Some(mesh) = polygon_face_mesh(face) else {
            return;
        };
        let mut edges = HashSet::new();
        for triangle in mesh.triangles {
            for (left, right) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                let edge = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                if !edges.insert(edge) {
                    continue;
                }
                let first = layer_point(preview, mesh.vertices[left]);
                let second = layer_point(preview, mesh.vertices[right]);
                unsafe {
                    CGMutablePath::move_to_point(Some(path), std::ptr::null(), first.x, first.y);
                    CGMutablePath::add_line_to_point(
                        Some(path),
                        std::ptr::null(),
                        second.x,
                        second.y,
                    );
                }
            }
        }
    }

    fn add_depth_face_mask(
        paths: &[CFRetained<CGMutablePath>],
        preview: &AVCaptureVideoPreviewLayer,
        face: &facefeature::FaceGeometry,
    ) {
        if paths.is_empty() {
            return;
        }
        let Some(mesh) = polygon_face_mesh(face) else {
            return;
        };
        let depths = mesh
            .vertices
            .iter()
            .map(|point| pseudo_face_depth(face, *point))
            .collect::<Vec<_>>();
        let mut edge_brightness = BTreeMap::<(usize, usize), (f64, usize)>::new();
        for triangle in &mesh.triangles {
            let brightness = triangle_surface_brightness(face, &mesh, &depths, *triangle);
            for (left, right) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                let edge = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                let accumulated = edge_brightness.entry(edge).or_insert((0.0, 0));
                accumulated.0 += brightness;
                accumulated.1 += 1;
            }
        }
        for ((left, right), (brightness_sum, triangle_count)) in edge_brightness {
            let brightness = brightness_sum / triangle_count as f64;
            let path = &paths[shade_band(brightness, paths.len())];
            let first = layer_point(preview, mesh.vertices[left]);
            let second = layer_point(preview, mesh.vertices[right]);
            unsafe {
                CGMutablePath::move_to_point(Some(path), std::ptr::null(), first.x, first.y);
                CGMutablePath::add_line_to_point(Some(path), std::ptr::null(), second.x, second.y);
            }
        }
    }

    fn triangle_surface_brightness(
        face: &facefeature::FaceGeometry,
        mesh: &PolygonFaceMesh,
        depths: &[f64],
        triangle: [usize; 3],
    ) -> f64 {
        let vertex = |index: usize| canonical_face_position(face, mesh.vertices[index]);
        let first = vertex(triangle[0]);
        let second = vertex(triangle[1]);
        let third = vertex(triangle[2]);
        let left = [
            second[0] - first[0],
            second[1] - first[1],
            second[2] - first[2],
        ];
        let right = [
            third[0] - first[0],
            third[1] - first[1],
            third[2] - first[2],
        ];
        let mut normal = [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ];
        if normal[2] < 0.0 {
            normal = [-normal[0], -normal[1], -normal[2]];
        }
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length <= f64::EPSILON {
            return 0.35;
        }
        normal = [normal[0] / length, normal[1] / length, normal[2] / length];
        normal = rotate_surface_normal(face, normal);
        let light = [-0.24, 0.18, 0.954];
        let diffuse =
            (normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]).clamp(0.0, 1.0);
        let visibility = ((normal[2] + 0.12) / 0.52).clamp(0.0, 1.0);
        let average_depth = (depths[triangle[0]] + depths[triangle[1]] + depths[triangle[2]]) / 3.0;
        (0.18 + visibility * (0.12 + diffuse * 0.55) + average_depth * 0.10).clamp(0.0, 1.0)
    }

    fn rotate_surface_normal(face: &facefeature::FaceGeometry, normal: [f64; 3]) -> [f64; 3] {
        let yaw = face.yaw_radians.unwrap_or(0.0);
        let pitch = face.pitch_radians.unwrap_or_else(|| {
            landmark_pitch_degrees(face)
                .map(f64::to_radians)
                .unwrap_or(0.0)
        });
        let roll = face.roll_radians.unwrap_or(0.0);

        let (yaw_sin, yaw_cos) = yaw.sin_cos();
        let yawed = [
            normal[0] * yaw_cos + normal[2] * yaw_sin,
            normal[1],
            -normal[0] * yaw_sin + normal[2] * yaw_cos,
        ];
        let (pitch_sin, pitch_cos) = pitch.sin_cos();
        let pitched = [
            yawed[0],
            yawed[1] * pitch_cos - yawed[2] * pitch_sin,
            yawed[1] * pitch_sin + yawed[2] * pitch_cos,
        ];
        let (roll_sin, roll_cos) = roll.sin_cos();
        [
            pitched[0] * roll_cos - pitched[1] * roll_sin,
            pitched[0] * roll_sin + pitched[1] * roll_cos,
            pitched[2],
        ]
    }

    fn pseudo_face_depth(face: &facefeature::FaceGeometry, point: Point) -> f64 {
        let bounds = face.bounding_box;
        if bounds.width <= f64::EPSILON || bounds.height <= f64::EPSILON {
            return 0.0;
        }
        let (normalized_x, normalized_y) = canonical_face_coordinates(face, point);
        let radial_squared = normalized_x * normalized_x + normalized_y * normalized_y;
        // A rational dome stays smooth beyond the nominal face ellipse. Unlike a hemisphere's
        // sqrt(max(0, ...)), it does not collapse profile-warped vertices onto a flat Z=0 sheet.
        let mut depth = 0.78 / (1.0 + radial_squared * 1.15);

        if let Some(nose) = landmark_center(face, LandmarkKind::Nose)
            .or_else(|| landmark_center(face, LandmarkKind::NoseCrest))
        {
            depth +=
                0.30 * feature_depth_weight(point, nose, bounds.width * 0.15, bounds.height * 0.20);
        }
        if let Some(mouth) = landmark_center(face, LandmarkKind::OuterLips) {
            depth += 0.07
                * feature_depth_weight(point, mouth, bounds.width * 0.23, bounds.height * 0.12);
        }
        for eye in [
            landmark_center(face, LandmarkKind::LeftEye),
            landmark_center(face, LandmarkKind::RightEye),
        ]
        .into_iter()
        .flatten()
        {
            depth -=
                0.06 * feature_depth_weight(point, eye, bounds.width * 0.16, bounds.height * 0.10);
        }
        depth.clamp(0.08, 1.0)
    }

    fn canonical_face_position(face: &facefeature::FaceGeometry, point: Point) -> [f64; 3] {
        let (x, y) = canonical_face_coordinates(face, point);
        [x, y, pseudo_face_depth(face, point) * 0.46]
    }

    fn canonical_face_coordinates(face: &facefeature::FaceGeometry, point: Point) -> (f64, f64) {
        let bounds = face.bounding_box;
        let center = Point {
            x: bounds.x + bounds.width * 0.5,
            y: bounds.y + bounds.height * 0.5,
        };
        let roll = face.roll_radians.unwrap_or(0.0);
        let (roll_sin, roll_cos) = roll.sin_cos();
        let offset_x = point.x - center.x;
        let offset_y = point.y - center.y;
        let unrolled_x = offset_x * roll_cos + offset_y * roll_sin;
        let unrolled_y = -offset_x * roll_sin + offset_y * roll_cos;
        (
            unrolled_x / (bounds.width * 0.54).max(f64::EPSILON),
            unrolled_y / (bounds.height * 0.58).max(f64::EPSILON),
        )
    }

    fn feature_depth_weight(point: Point, center: Point, radius_x: f64, radius_y: f64) -> f64 {
        if radius_x <= f64::EPSILON || radius_y <= f64::EPSILON {
            return 0.0;
        }
        let x = (point.x - center.x) / radius_x;
        let y = (point.y - center.y) / radius_y;
        (-0.5 * (x * x + y * y)).exp()
    }

    fn shade_band(brightness: f64, layer_count: usize) -> usize {
        if layer_count <= 1 {
            return 0;
        }
        ((brightness.clamp(0.0, 1.0) * (layer_count - 1) as f64).round() as usize)
            .min(layer_count - 1)
    }

    #[derive(Debug)]
    struct PolygonFaceMesh {
        vertices: Vec<Point>,
        triangles: Vec<[usize; 3]>,
    }

    fn polygon_face_mesh(face: &facefeature::FaceGeometry) -> Option<PolygonFaceMesh> {
        let bounds = face.bounding_box;
        if bounds.width <= f64::EPSILON || bounds.height <= f64::EPSILON {
            return None;
        }
        let mut vertices = face
            .landmarks
            .iter()
            .find(|region| region.kind == LandmarkKind::AllPoints)
            .map(|region| region.points.clone())
            .unwrap_or_else(|| {
                face.landmarks
                    .iter()
                    .filter(|region| region.kind != LandmarkKind::AllPoints)
                    .flat_map(|region| region.points.iter().copied())
                    .collect()
            });
        // `allPoints` is convenient for the dense base mesh, but at steep yaw it can omit the
        // outermost samples used by Vision's dedicated eye/eyebrow regions. Keep those feature
        // points as exact mesh anchors so the generated surface meets the landmark overlay.
        vertices.extend(
            face.landmarks
                .iter()
                .filter(|region| {
                    matches!(
                        region.kind,
                        LandmarkKind::LeftEye
                            | LandmarkKind::RightEye
                            | LandmarkKind::LeftEyebrow
                            | LandmarkKind::RightEyebrow
                    )
                })
                .flat_map(|region| region.points.iter().copied()),
        );
        deduplicate_mesh_vertices(&mut vertices, bounds.width, bounds.height);
        add_forehead_mesh_vertices(face, &mut vertices);
        apply_yaw_visibility_warp(face, &mut vertices);
        deduplicate_mesh_vertices(&mut vertices, bounds.width, bounds.height);
        interpolate_uniform_mesh(&mut vertices, bounds);
        deduplicate_mesh_vertices(&mut vertices, bounds.width, bounds.height);
        if vertices.len() < 3 {
            return None;
        }

        let local_vertices = vertices
            .iter()
            .map(|point| Point {
                x: (point.x - bounds.x) / bounds.width,
                y: (point.y - bounds.y) / bounds.height,
            })
            .collect::<Vec<_>>();
        let triangles = delaunay_triangles(&local_vertices)
            .into_iter()
            .filter(|triangle| !triangle_is_feature_hole(face, &vertices, *triangle))
            .collect::<Vec<_>>();
        (!triangles.is_empty()).then_some(PolygonFaceMesh {
            vertices,
            triangles,
        })
    }

    fn deduplicate_mesh_vertices(vertices: &mut Vec<Point>, width: f64, height: f64) {
        let mut unique = Vec::<Point>::with_capacity(vertices.len());
        for point in vertices.drain(..) {
            let duplicate = unique.iter().any(|candidate| {
                let dx = (candidate.x - point.x) / width;
                let dy = (candidate.y - point.y) / height;
                dx * dx + dy * dy < 0.000_025
            });
            if !duplicate {
                unique.push(point);
            }
        }
        *vertices = unique;
    }

    fn add_forehead_mesh_vertices(face: &facefeature::FaceGeometry, vertices: &mut Vec<Point>) {
        let bounds = face.bounding_box;
        let left_eye = landmark_center(face, LandmarkKind::LeftEye);
        let right_eye = landmark_center(face, LandmarkKind::RightEye);
        let (origin, horizontal, vertical) = match left_eye.zip(right_eye) {
            Some((left, right)) => {
                let mut horizontal = Point {
                    x: right.x - left.x,
                    y: right.y - left.y,
                };
                let length = horizontal.x.hypot(horizontal.y);
                if length <= f64::EPSILON {
                    return;
                }
                horizontal.x /= length;
                horizontal.y /= length;
                let mut vertical = Point {
                    x: -horizontal.y,
                    y: horizontal.x,
                };
                if vertical.y < 0.0 {
                    vertical.x = -vertical.x;
                    vertical.y = -vertical.y;
                }
                (
                    Point {
                        x: (left.x + right.x) * 0.5,
                        y: (left.y + right.y) * 0.5,
                    },
                    horizontal,
                    vertical,
                )
            }
            None => (
                Point {
                    x: bounds.x + bounds.width * 0.5,
                    y: bounds.y + bounds.height * 0.58,
                },
                Point { x: 1.0, y: 0.0 },
                Point { x: 0.0, y: 1.0 },
            ),
        };

        let horizontal_projection = |point: Point| {
            (point.x - origin.x) * horizontal.x + (point.y - origin.y) * horizontal.y
        };
        let vertical_projection =
            |point: Point| (point.x - origin.x) * vertical.x + (point.y - origin.y) * vertical.y;
        let contour = face
            .landmarks
            .iter()
            .find(|region| region.kind == LandmarkKind::FaceContour)
            .map(|region| region.points.as_slice())
            .unwrap_or_default();
        let Some(upward_ray) = ray_distance_to_bounds(origin, vertical, bounds) else {
            return;
        };
        let eye_to_chin = contour
            .iter()
            .copied()
            .map(vertical_projection)
            .reduce(f64::min)
            .filter(|distance| *distance < 0.0)
            .map(|distance| -distance)
            .unwrap_or(bounds.height * 0.5);
        let anatomical_forehead =
            (eye_to_chin * 0.62).clamp(bounds.height * 0.18, bounds.height * 0.38);
        let upward_space = upward_ray.min(anatomical_forehead);

        let temple_projections = contour
            .first()
            .copied()
            .zip(contour.last().copied())
            .map(|(first, last)| [horizontal_projection(first), horizontal_projection(last)]);
        let (left_extent, right_extent) = temple_projections
            .map(|values| (values[0].min(values[1]), values[0].max(values[1])))
            .unwrap_or((f64::INFINITY, f64::NEG_INFINITY));
        let (left_extent, right_extent) = if left_extent.is_finite()
            && right_extent.is_finite()
            && right_extent - left_extent > f64::EPSILON
        {
            (left_extent, right_extent)
        } else {
            (-bounds.width * 0.45, bounds.width * 0.45)
        };
        let forehead_center = (left_extent + right_extent) * 0.5;
        const FOREHEAD_ROWS: [(f64, f64, usize); 5] = [
            (0.06, 0.98, 11),
            (0.27, 0.92, 11),
            (0.48, 0.83, 10),
            (0.69, 0.71, 9),
            (0.90, 0.57, 7),
        ];
        for (height_fraction, width_fraction, columns) in FOREHEAD_ROWS {
            let row_left = forehead_center + (left_extent - forehead_center) * width_fraction;
            let row_right = forehead_center + (right_extent - forehead_center) * width_fraction;
            for column in 0..columns {
                let horizontal_offset = if columns == 1 {
                    forehead_center
                } else {
                    row_left + (row_right - row_left) * column as f64 / (columns - 1) as f64
                };
                vertices.push(Point {
                    x: origin.x
                        + horizontal.x * horizontal_offset
                        + vertical.x * upward_space * height_fraction,
                    y: origin.y
                        + horizontal.y * horizontal_offset
                        + vertical.y * upward_space * height_fraction,
                });
            }
        }
    }

    fn ray_distance_to_bounds(
        origin: Point,
        direction: Point,
        bounds: facefeature::BoundingBox,
    ) -> Option<f64> {
        let mut distances = Vec::with_capacity(2);
        if direction.x > f64::EPSILON {
            distances.push((bounds.x + bounds.width - origin.x) / direction.x);
        } else if direction.x < -f64::EPSILON {
            distances.push((bounds.x - origin.x) / direction.x);
        }
        if direction.y > f64::EPSILON {
            distances.push((bounds.y + bounds.height - origin.y) / direction.y);
        } else if direction.y < -f64::EPSILON {
            distances.push((bounds.y - origin.y) / direction.y);
        }
        distances
            .into_iter()
            .filter(|distance| distance.is_finite() && *distance > 0.0)
            .reduce(f64::min)
    }

    fn apply_yaw_visibility_warp(face: &facefeature::FaceGeometry, vertices: &mut [Point]) {
        const WARP_START_DEGREES: f64 = 18.0;
        const WARP_FULL_DEGREES: f64 = 45.0;
        const FRONTAL_MARGIN: f64 = 0.52;
        const PROFILE_MARGIN: f64 = 0.16;

        let Some(yaw_degrees) = face.yaw_radians.map(f64::to_degrees) else {
            return;
        };
        let strength = ((yaw_degrees.abs() - WARP_START_DEGREES)
            / (WARP_FULL_DEGREES - WARP_START_DEGREES))
            .clamp(0.0, 1.0);
        if strength <= f64::EPSILON {
            return;
        }
        let axis = yaw_visibility_axis(face);
        if axis.len() < 2 {
            return;
        }
        let margin = face.bounding_box.width
            * (FRONTAL_MARGIN + (PROFILE_MARGIN - FRONTAL_MARGIN) * strength);
        let eye_envelope = eye_feature_envelope(face);
        for vertex in vertices {
            let center_x = interpolate_axis_x(&axis, vertex.y);
            if yaw_degrees > 0.0 {
                let mut limit = center_x + margin;
                if let Some(envelope) = eye_envelope {
                    let influence = envelope.vertical_influence(vertex.y, face.bounding_box.height);
                    let feature_limit = envelope.max_x + face.bounding_box.width * 0.018;
                    let expanded_limit = limit.max(feature_limit);
                    limit += (expanded_limit - limit) * influence;
                }
                if vertex.x > limit {
                    vertex.x = limit + (vertex.x - limit) * (1.0 - strength);
                }
            } else {
                let mut limit = center_x - margin;
                if let Some(envelope) = eye_envelope {
                    let influence = envelope.vertical_influence(vertex.y, face.bounding_box.height);
                    let feature_limit = envelope.min_x - face.bounding_box.width * 0.018;
                    let expanded_limit = limit.min(feature_limit);
                    limit += (expanded_limit - limit) * influence;
                }
                if vertex.x < limit {
                    vertex.x = limit + (vertex.x - limit) * (1.0 - strength);
                }
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct EyeFeatureEnvelope {
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    }

    impl EyeFeatureEnvelope {
        fn vertical_influence(self, y: f64, face_height: f64) -> f64 {
            let distance = if y < self.min_y {
                self.min_y - y
            } else if y > self.max_y {
                y - self.max_y
            } else {
                return 1.0;
            };
            let taper = (face_height * 0.07).max(f64::EPSILON);
            (1.0 - distance / taper).clamp(0.0, 1.0)
        }
    }

    fn eye_feature_envelope(face: &facefeature::FaceGeometry) -> Option<EyeFeatureEnvelope> {
        let mut points = face
            .landmarks
            .iter()
            .filter(|region| {
                matches!(
                    region.kind,
                    LandmarkKind::LeftEye
                        | LandmarkKind::RightEye
                        | LandmarkKind::LeftEyebrow
                        | LandmarkKind::RightEyebrow
                )
            })
            .flat_map(|region| region.points.iter().copied());
        let first = points.next()?;
        Some(points.fold(
            EyeFeatureEnvelope {
                min_x: first.x,
                max_x: first.x,
                min_y: first.y,
                max_y: first.y,
            },
            |mut envelope, point| {
                envelope.min_x = envelope.min_x.min(point.x);
                envelope.max_x = envelope.max_x.max(point.x);
                envelope.min_y = envelope.min_y.min(point.y);
                envelope.max_y = envelope.max_y.max(point.y);
                envelope
            },
        ))
    }

    fn yaw_visibility_axis(face: &facefeature::FaceGeometry) -> Vec<Point> {
        let bounds = face.bounding_box;
        let eyes = landmark_center(face, LandmarkKind::LeftEye)
            .zip(landmark_center(face, LandmarkKind::RightEye))
            .map(|(left, right)| Point {
                x: (left.x + right.x) * 0.5,
                y: (left.y + right.y) * 0.5,
            });
        let nose = landmark_center(face, LandmarkKind::Nose)
            .or_else(|| landmark_center(face, LandmarkKind::NoseCrest));
        let mouth = landmark_center(face, LandmarkKind::OuterLips);
        let chin = face
            .landmarks
            .iter()
            .find(|region| region.kind == LandmarkKind::FaceContour)
            .and_then(|region| {
                region
                    .points
                    .iter()
                    .copied()
                    .min_by(|left, right| left.y.total_cmp(&right.y))
            });
        let top_x = eyes
            .or(nose)
            .map(|point| point.x)
            .unwrap_or(bounds.x + bounds.width * 0.5);
        let mut axis = vec![Point {
            x: top_x,
            y: bounds.y + bounds.height,
        }];
        axis.extend([eyes, nose, mouth, chin].into_iter().flatten());
        axis.sort_by(|left, right| left.y.total_cmp(&right.y));
        axis.dedup_by(|left, right| (left.y - right.y).abs() < 1e-8);
        axis
    }

    fn interpolate_axis_x(axis: &[Point], y: f64) -> f64 {
        if y <= axis[0].y {
            return axis[0].x;
        }
        if y >= axis[axis.len() - 1].y {
            return axis[axis.len() - 1].x;
        }
        axis.windows(2)
            .find_map(|segment| {
                let [lower, upper] = segment else {
                    return None;
                };
                (y >= lower.y && y <= upper.y).then(|| {
                    let progress = (y - lower.y) / (upper.y - lower.y);
                    lower.x + (upper.x - lower.x) * progress
                })
            })
            .unwrap_or(axis[axis.len() - 1].x)
    }

    fn interpolate_uniform_mesh(vertices: &mut Vec<Point>, bounds: facefeature::BoundingBox) {
        const TARGET_EDGE_LENGTH: f64 = 0.075;
        const MAX_REFINEMENT_PASSES: usize = 6;
        const MAX_MESH_VERTICES: usize = 600;
        if vertices.len() < 3 || bounds.width <= f64::EPSILON || bounds.height <= f64::EPSILON {
            return;
        }
        for _ in 0..MAX_REFINEMENT_PASSES {
            let local_vertices = vertices
                .iter()
                .map(|point| Point {
                    x: (point.x - bounds.x) / bounds.width,
                    y: (point.y - bounds.y) / bounds.height,
                })
                .collect::<Vec<_>>();
            let triangles = delaunay_triangles(&local_vertices);
            let mut long_edges = BTreeSet::new();
            for triangle in triangles {
                for (left, right) in [
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                ] {
                    if local_vertices[left].distance(local_vertices[right]) <= TARGET_EDGE_LENGTH {
                        continue;
                    }
                    long_edges.insert(if left < right {
                        (left, right)
                    } else {
                        (right, left)
                    });
                }
            }
            if long_edges.is_empty() || vertices.len() >= MAX_MESH_VERTICES {
                break;
            }
            let remaining = MAX_MESH_VERTICES - vertices.len();
            let interpolated = long_edges
                .into_iter()
                .take(remaining)
                .map(|(left, right)| Point {
                    x: (vertices[left].x + vertices[right].x) * 0.5,
                    y: (vertices[left].y + vertices[right].y) * 0.5,
                })
                .collect::<Vec<_>>();
            vertices.extend(interpolated);
            deduplicate_mesh_vertices(vertices, bounds.width, bounds.height);
        }
    }

    fn delaunay_triangles(points: &[Point]) -> Vec<[usize; 3]> {
        if points.len() < 3 {
            return Vec::new();
        }
        let original_count = points.len();
        let mut work = points.to_vec();
        work.extend([
            Point { x: -10.0, y: -10.0 },
            Point { x: 10.0, y: -10.0 },
            Point { x: 0.0, y: 10.0 },
        ]);
        let mut triangles = vec![[original_count, original_count + 1, original_count + 2]];

        for point_index in 0..original_count {
            let mut edge_counts = BTreeMap::<(usize, usize), usize>::new();
            let mut retained = Vec::with_capacity(triangles.len());
            for triangle in triangles {
                if circumcircle_contains(&work, triangle, work[point_index]) {
                    for (left, right) in [
                        (triangle[0], triangle[1]),
                        (triangle[1], triangle[2]),
                        (triangle[2], triangle[0]),
                    ] {
                        let edge = if left < right {
                            (left, right)
                        } else {
                            (right, left)
                        };
                        *edge_counts.entry(edge).or_default() += 1;
                    }
                } else {
                    retained.push(triangle);
                }
            }
            triangles = retained;
            triangles.extend(
                edge_counts
                    .into_iter()
                    .filter(|(_, count)| *count == 1)
                    .filter_map(|((left, right), _)| {
                        (triangle_area(work[left], work[right], work[point_index]).abs() > 1e-10)
                            .then_some([left, right, point_index])
                    }),
            );
        }

        triangles
            .into_iter()
            .filter(|triangle| triangle.iter().all(|index| *index < original_count))
            .collect()
    }

    fn circumcircle_contains(points: &[Point], triangle: [usize; 3], point: Point) -> bool {
        let [a, b, c] = triangle.map(|index| points[index]);
        let denominator = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
        if denominator.abs() <= 1e-12 {
            return false;
        }
        let a_squared = a.x * a.x + a.y * a.y;
        let b_squared = b.x * b.x + b.y * b.y;
        let c_squared = c.x * c.x + c.y * c.y;
        let center = Point {
            x: (a_squared * (b.y - c.y) + b_squared * (c.y - a.y) + c_squared * (a.y - b.y))
                / denominator,
            y: (a_squared * (c.x - b.x) + b_squared * (a.x - c.x) + c_squared * (b.x - a.x))
                / denominator,
        };
        let radius_squared = (center.x - a.x).powi(2) + (center.y - a.y).powi(2);
        let distance_squared = (center.x - point.x).powi(2) + (center.y - point.y).powi(2);
        distance_squared <= radius_squared + 1e-10
    }

    fn triangle_area(a: Point, b: Point, c: Point) -> f64 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }

    fn triangle_is_feature_hole(
        face: &facefeature::FaceGeometry,
        vertices: &[Point],
        triangle: [usize; 3],
    ) -> bool {
        let center = triangle.into_iter().map(|index| vertices[index]).fold(
            Point { x: 0.0, y: 0.0 },
            |sum, point| Point {
                x: sum.x + point.x / 3.0,
                y: sum.y + point.y / 3.0,
            },
        );
        [
            LandmarkKind::LeftEye,
            LandmarkKind::RightEye,
            LandmarkKind::InnerLips,
        ]
        .into_iter()
        .filter_map(|kind| face.landmarks.iter().find(|region| region.kind == kind))
        .any(|region| point_in_polygon(center, &region.points))
    }

    fn point_in_polygon(point: Point, polygon: &[Point]) -> bool {
        if polygon.len() < 3 {
            return false;
        }
        let mut inside = false;
        let mut previous = polygon.len() - 1;
        for current in 0..polygon.len() {
            let a = polygon[current];
            let b = polygon[previous];
            if (a.y > point.y) != (b.y > point.y)
                && point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x
            {
                inside = !inside;
            }
            previous = current;
        }
        inside
    }

    fn landmark_center(face: &facefeature::FaceGeometry, kind: LandmarkKind) -> Option<Point> {
        let points = &face
            .landmarks
            .iter()
            .find(|region| region.kind == kind)?
            .points;
        (!points.is_empty()).then(|| {
            let sum = points
                .iter()
                .fold(Point { x: 0.0, y: 0.0 }, |sum, point| Point {
                    x: sum.x + point.x,
                    y: sum.y + point.y,
                });
            Point {
                x: sum.x / points.len() as f64,
                y: sum.y / points.len() as f64,
            }
        })
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

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum FaceMaskMode {
        #[default]
        None,
        Polygon,
        Depth,
    }

    #[derive(Debug, Default, PartialEq)]
    struct CameraOptions {
        face_id_model: Option<PathBuf>,
        face_id_database: Option<PathBuf>,
        face_id_mode: Option<FaceIdDatabaseMode>,
        capture_requested: bool,
        capture_target: Option<FaceCaptureTarget>,
        benchmark_iterations: Option<usize>,
        face_mask: FaceMaskMode,
        mask_only: bool,
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
                "--face-mask" => {
                    let value = arguments
                        .next()
                        .ok_or("--face-mask requires polygon, depth, or none")?;
                    options.face_mask = match value.as_str() {
                        "polygon" => FaceMaskMode::Polygon,
                        "depth" => FaceMaskMode::Depth,
                        "none" => FaceMaskMode::None,
                        _ => {
                            return Err("--face-mask requires polygon, depth, or none".to_owned());
                        }
                    };
                }
                "--mask-only" => options.mask_only = true,
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
            "facefeature-camera\n\nUSAGE:\n    facefeature-camera [OPTIONS]\n    facefeature-camera --face-id [OPTIONS]\n    facefeature-camera --read-only [OPTIONS]\n    facefeature-camera --capture (--name NAME | --person ID) [OPTIONS]\n    facefeature-camera --benchmark [OPTIONS]\n\nOPTIONS:\n    --face-id                 Automatic matching and enrollment\n    --read-only               Match from SQLite without inserting or updating rows\n    --capture                 Guided center/left/right/up/down enrollment\n    --name NAME               Create a new named identity during guided capture\n    --person ID               Replace/add guided samples for an existing identity\n    --face-mask MODE          Face overlay: polygon, depth, or none (default)\n    --mask-only               Hide the camera image; retain generated overlays\n    --benchmark               Benchmark headless SFace/Core ML inference\n    --benchmark-iterations N  Timed iterations (default: 100)\n    --face-id-model PATH      Use a custom SFace ONNX model\n    --face-id-db PATH         Use a custom SQLite identity gallery path\n    -h, --help                Print this help"
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
        let delegate = AppDelegate::new(mtm, face_id, options.face_mask, options.mask_only);
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
        fn polygon_face_mask_is_an_independent_camera_option() {
            let options = parse_options(["--face-mask".to_owned(), "polygon".to_owned()]).unwrap();
            assert_eq!(options.face_mask, FaceMaskMode::Polygon);
            assert!(options.face_id_model.is_none());
            assert!(options.face_id_database.is_none());
        }

        #[test]
        fn mask_only_hides_only_the_camera_presentation() {
            let options = parse_options([
                "--mask-only".to_owned(),
                "--face-mask".to_owned(),
                "polygon".to_owned(),
                "--face-id".to_owned(),
            ])
            .unwrap();
            assert!(options.mask_only);
            assert_eq!(options.face_mask, FaceMaskMode::Polygon);
            assert!(options.face_id_model.is_some());
            assert!(options.face_id_database.is_some());
        }

        #[test]
        fn depth_face_mask_is_available_with_mask_only() {
            let options = parse_options([
                "--face-mask".to_owned(),
                "depth".to_owned(),
                "--mask-only".to_owned(),
            ])
            .unwrap();
            assert_eq!(options.face_mask, FaceMaskMode::Depth);
            assert!(options.mask_only);
        }

        #[test]
        fn face_mask_rejects_unknown_modes() {
            let error = parse_options(["--face-mask".to_owned(), "metal".to_owned()]).unwrap_err();
            assert_eq!(error, "--face-mask requires polygon, depth, or none");
        }

        #[test]
        fn pseudo_depth_places_the_nose_in_front_of_the_face_edge() {
            let bounds = facefeature::BoundingBox {
                x: 0.2,
                y: 0.1,
                width: 0.6,
                height: 0.8,
            };
            let landmarks = vec![
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::FaceContour,
                    points: vec![Point { x: 0.2, y: 0.5 }, Point { x: 0.5, y: 0.1 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::Nose,
                    points: vec![Point { x: 0.5, y: 0.5 }],
                },
            ];
            let mut face = facefeature::FaceGeometry {
                confidence: 0.99,
                landmark_confidence: 0.99,
                bounding_box: bounds,
                roll_radians: None,
                yaw_radians: Some(0.0),
                pitch_radians: None,
                measurements: facefeature::FaceGeometry::calculate_measurements(bounds, &landmarks),
                landmarks,
            };
            let nose_depth = pseudo_face_depth(&face, Point { x: 0.5, y: 0.5 });
            let edge_depth = pseudo_face_depth(&face, Point { x: 0.2, y: 0.5 });
            assert!(nose_depth > edge_depth + 0.5);
            let outside_depth = pseudo_face_depth(&face, Point { x: 0.08, y: 0.5 });
            assert!(outside_depth >= 0.08);
            face.yaw_radians = Some((-45.0_f64).to_radians());
            let profile_edge_depth = pseudo_face_depth(&face, Point { x: 0.2, y: 0.5 });
            assert!((profile_edge_depth - edge_depth).abs() < 1e-12);
            assert_eq!(shade_band(0.0, DEPTH_LAYER_COUNT), 0);
            assert_eq!(shade_band(1.0, DEPTH_LAYER_COUNT), DEPTH_LAYER_COUNT - 1);
            assert_eq!(shade_band(0.5, DEPTH_LAYER_COUNT), 4);
        }

        #[test]
        fn polygon_mesh_densifies_the_contour_with_a_forehead() {
            let bounds = facefeature::BoundingBox {
                x: 0.2,
                y: 0.1,
                width: 0.5,
                height: 0.7,
            };
            let landmarks = vec![
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::FaceContour,
                    points: vec![
                        Point { x: 0.23, y: 0.55 },
                        Point { x: 0.27, y: 0.24 },
                        Point { x: 0.45, y: 0.13 },
                        Point { x: 0.63, y: 0.24 },
                        // Rotated/smoothed landmark geometry can legitimately extend beyond the
                        // current axis-aligned box (whose right edge is 0.70).
                        Point { x: 0.72, y: 0.55 },
                    ],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::Nose,
                    points: vec![Point { x: 0.45, y: 0.43 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::LeftEye,
                    points: vec![Point { x: 0.35, y: 0.74 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::RightEye,
                    points: vec![Point { x: 0.55, y: 0.74 }],
                },
            ];
            let face = facefeature::FaceGeometry {
                confidence: 0.99,
                landmark_confidence: 0.99,
                bounding_box: bounds,
                roll_radians: None,
                yaw_radians: None,
                pitch_radians: None,
                measurements: facefeature::FaceGeometry::calculate_measurements(bounds, &landmarks),
                landmarks,
            };

            let mesh = polygon_face_mesh(&face).unwrap();
            assert!(mesh.vertices.len() > landmarks_point_count(&face) + 48);
            assert!(mesh.triangles.len() > 20);
            assert!(
                mesh.vertices
                    .iter()
                    .any(|point| point.y > bounds.y + bounds.height * 0.9)
            );
            assert!(
                mesh.triangles
                    .iter()
                    .all(|triangle| triangle.iter().all(|index| *index < mesh.vertices.len()))
            );
            assert!(
                mesh.vertices
                    .iter()
                    .all(|point| (0.23..=0.72).contains(&point.x))
            );
            assert!(
                mesh.vertices
                    .iter()
                    .any(|point| point.x > bounds.x + bounds.width)
            );
            let longest_edge = mesh
                .triangles
                .iter()
                .flat_map(|triangle| {
                    [
                        (triangle[0], triangle[1]),
                        (triangle[1], triangle[2]),
                        (triangle[2], triangle[0]),
                    ]
                })
                .map(|(left, right)| {
                    let left = Point {
                        x: (mesh.vertices[left].x - bounds.x) / bounds.width,
                        y: (mesh.vertices[left].y - bounds.y) / bounds.height,
                    };
                    let right = Point {
                        x: (mesh.vertices[right].x - bounds.x) / bounds.width,
                        y: (mesh.vertices[right].y - bounds.y) / bounds.height,
                    };
                    left.distance(right)
                })
                .fold(0.0_f64, f64::max);
            assert!(longest_edge <= 0.085, "longest edge was {longest_edge}");
            assert!(mesh.vertices.len() <= 600);
        }

        #[test]
        fn delaunay_mesh_triangulates_a_square() {
            let triangles = delaunay_triangles(&[
                Point { x: 0.0, y: 0.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 1.0, y: 1.0 },
                Point { x: 0.0, y: 1.0 },
            ]);
            assert_eq!(triangles.len(), 2);
        }

        #[test]
        fn strong_yaw_warps_only_the_self_occluded_mesh_side() {
            let bounds = facefeature::BoundingBox {
                x: 0.2,
                y: 0.1,
                width: 0.5,
                height: 0.7,
            };
            let landmarks = vec![
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::FaceContour,
                    points: vec![Point { x: 0.23, y: 0.55 }, Point { x: 0.45, y: 0.13 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::LeftEye,
                    points: vec![Point { x: 0.35, y: 0.65 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::RightEye,
                    points: vec![Point { x: 0.55, y: 0.65 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::Nose,
                    points: vec![Point { x: 0.45, y: 0.45 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::OuterLips,
                    points: vec![Point { x: 0.45, y: 0.30 }],
                },
            ];
            let face = facefeature::FaceGeometry {
                confidence: 0.99,
                landmark_confidence: 0.99,
                bounding_box: bounds,
                roll_radians: None,
                yaw_radians: Some(45.0_f64.to_radians()),
                pitch_radians: None,
                measurements: facefeature::FaceGeometry::calculate_measurements(bounds, &landmarks),
                landmarks,
            };
            let mut vertices = vec![Point { x: 0.67, y: 0.55 }, Point { x: 0.23, y: 0.55 }];
            apply_yaw_visibility_warp(&face, &mut vertices);
            assert!(vertices[0].x <= 0.531);
            assert!((vertices[1].x - 0.23).abs() < 1e-12);

            let mut opposite_face = face;
            opposite_face.yaw_radians = Some((-45.0_f64).to_radians());
            let mut opposite_vertices =
                vec![Point { x: 0.67, y: 0.55 }, Point { x: 0.23, y: 0.55 }];
            apply_yaw_visibility_warp(&opposite_face, &mut opposite_vertices);
            assert!((opposite_vertices[0].x - 0.67).abs() < 1e-12);
            assert!(opposite_vertices[1].x >= 0.369);
        }

        #[test]
        fn yaw_warp_preserves_eye_and_eyebrow_envelope_without_restoring_the_wing() {
            let bounds = facefeature::BoundingBox {
                x: 0.2,
                y: 0.1,
                width: 0.5,
                height: 0.7,
            };
            let landmarks = vec![
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::FaceContour,
                    points: vec![Point { x: 0.23, y: 0.55 }, Point { x: 0.45, y: 0.13 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::LeftEye,
                    points: vec![Point { x: 0.31, y: 0.65 }, Point { x: 0.37, y: 0.66 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::RightEye,
                    points: vec![Point { x: 0.53, y: 0.66 }, Point { x: 0.59, y: 0.65 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::LeftEyebrow,
                    points: vec![Point { x: 0.30, y: 0.71 }, Point { x: 0.38, y: 0.72 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::RightEyebrow,
                    points: vec![Point { x: 0.52, y: 0.72 }, Point { x: 0.60, y: 0.71 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::Nose,
                    points: vec![Point { x: 0.45, y: 0.45 }],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::OuterLips,
                    points: vec![Point { x: 0.45, y: 0.30 }],
                },
            ];
            let mut face = facefeature::FaceGeometry {
                confidence: 0.99,
                landmark_confidence: 0.99,
                bounding_box: bounds,
                roll_radians: None,
                yaw_radians: Some((-45.0_f64).to_radians()),
                pitch_radians: None,
                measurements: facefeature::FaceGeometry::calculate_measurements(bounds, &landmarks),
                landmarks,
            };

            let mut left_profile_vertices =
                vec![Point { x: 0.30, y: 0.71 }, Point { x: 0.23, y: 0.50 }];
            apply_yaw_visibility_warp(&face, &mut left_profile_vertices);
            assert!((left_profile_vertices[0].x - 0.30).abs() < 1e-12);
            assert!(left_profile_vertices[1].x >= 0.36);

            face.yaw_radians = Some(45.0_f64.to_radians());
            let mut right_profile_vertices =
                vec![Point { x: 0.60, y: 0.71 }, Point { x: 0.67, y: 0.50 }];
            apply_yaw_visibility_warp(&face, &mut right_profile_vertices);
            assert!((right_profile_vertices[0].x - 0.60).abs() < 1e-12);
            assert!(right_profile_vertices[1].x <= 0.54);
        }

        #[test]
        fn rolled_forehead_uses_temples_instead_of_wide_jaw_points() {
            let bounds = facefeature::BoundingBox {
                x: 0.2,
                y: 0.1,
                width: 0.6,
                height: 0.8,
            };
            let angle = 30.0_f64.to_radians();
            let horizontal = Point {
                x: angle.cos(),
                y: angle.sin(),
            };
            let vertical = Point {
                x: -angle.sin(),
                y: angle.cos(),
            };
            let origin = Point { x: 0.5, y: 0.57 };
            let offset = |across: f64, down: f64| Point {
                x: origin.x + horizontal.x * across - vertical.x * down,
                y: origin.y + horizontal.y * across - vertical.y * down,
            };
            let landmarks = vec![
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::FaceContour,
                    points: vec![
                        offset(-0.16, 0.0),
                        offset(-0.28, 0.24),
                        offset(0.0, 0.40),
                        offset(0.28, 0.24),
                        offset(0.16, 0.0),
                    ],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::LeftEye,
                    points: vec![offset(-0.08, 0.0)],
                },
                facefeature::LandmarkRegion {
                    kind: LandmarkKind::RightEye,
                    points: vec![offset(0.08, 0.0)],
                },
            ];
            let face = facefeature::FaceGeometry {
                confidence: 0.99,
                landmark_confidence: 0.99,
                bounding_box: bounds,
                roll_radians: Some(angle),
                yaw_radians: Some(0.0),
                pitch_radians: None,
                measurements: facefeature::FaceGeometry::calculate_measurements(bounds, &landmarks),
                landmarks,
            };
            let mut forehead = Vec::new();
            add_forehead_mesh_vertices(&face, &mut forehead);
            assert!(!forehead.is_empty());
            assert!(forehead.iter().all(|point| {
                let projection =
                    (point.x - origin.x) * horizontal.x + (point.y - origin.y) * horizontal.y;
                (-0.161..=0.161).contains(&projection)
            }));
            let maximum_height = forehead
                .iter()
                .map(|point| (point.x - origin.x) * vertical.x + (point.y - origin.y) * vertical.y)
                .fold(0.0_f64, f64::max);
            assert!(maximum_height <= 0.40 * 0.62 + 1e-12);
        }

        fn landmarks_point_count(face: &facefeature::FaceGeometry) -> usize {
            face.landmarks
                .iter()
                .map(|region| region.points.len())
                .sum()
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
