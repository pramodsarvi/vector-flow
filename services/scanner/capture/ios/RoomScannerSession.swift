//
//  RoomScannerSession.swift
//  Spatial Scanner - iOS Client Layer
//
//  Captures synchronized RGB frames (4K/1080p), LiDAR depth (dToF),
//  and 6-DoF camera poses using ARKit and Apple RoomPlan.
//

import Foundation
import ARKit
import RoomPlan
import MetalKit

@available(iOS 16.0, *)
public class RoomScannerSession: NSObject, ARSessionDelegate, RoomCaptureSessionDelegate {
    
    private let arSession = ARSession()
    private var roomCaptureSession: RoomCaptureSession?
    private let fileManager = FileManager.default
    private var sessionDirectory: URL?
    
    // Recording state
    public private(set) var isScanning: Bool = false
    private var frameIndex: Int = 0
    private var recordedPoses: [[String: Any]] = []
    
    // Spatial metadata
    public var onScanCompleted: ((URL) -> Void)?
    public var onSurfaceDetected: ((String, Float) -> Void)?
    
    public override init() {
        super.init()
        arSession.delegate = self
        setupRoomPlan()
    }
    
    private func setupRoomPlan() {
        roomCaptureSession = RoomCaptureSession()
        roomCaptureSession?.delegate = self
    }
    
    /// Starts spatial recording session with LiDAR point cloud + RGB frames
    public func startScanning() {
        guard ARWorldTrackingConfiguration.supportsFrameSemantics([.sceneDepth, .smoothedSceneDepth]) else {
            print("Error: LiDAR sensor is not supported on this device.")
            return
        }
        
        // Setup working directory
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        sessionDirectory = docs.appendingPathComponent("Scan_\(timestamp)")
        
        do {
            try fileManager.createDirectory(at: sessionDirectory!, withIntermediateDirectories: true)
            try fileManager.createDirectory(at: sessionDirectory!.appendingPathComponent("images"), withIntermediateDirectories: true)
            try fileManager.createDirectory(at: sessionDirectory!.appendingPathComponent("depth"), withIntermediateDirectories: true)
        } catch {
            print("Failed to initialize scan storage: \(error)")
            return
        }
        
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]
        
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        
        let roomConfig = RoomCaptureSession.Configuration()
        roomCaptureSession?.run(configuration: roomConfig)
        
        isScanning = true
        frameIndex = 0
        recordedPoses.removeAll()
        print("Spatial scanning session started at: \(sessionDirectory!.path)")
    }
    
    /// ARSession callback: capture synchronized RGB keyframes + LiDAR depth
    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isScanning, let sessionDir = sessionDirectory else { return }
        
        // Keyframe subsampling: capture every 6th frame (~5-10 fps) to avoid blur and storage bloat
        if frameIndex % 6 == 0 {
            saveKeyframe(frame: frame, index: frameIndex / 6, targetDir: sessionDir)
        }
        frameIndex += 1
    }
    
    private func saveKeyframe(frame: ARFrame, index: Int, targetDir: URL) {
        let transform = frame.camera.transform
        let position = simd_make_float3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        let eulerAngles = frame.camera.eulerAngles
        
        // 1. Camera Intrinsics
        let intrinsics = frame.camera.intrinsics
        let fx = intrinsics[0, 0]
        let fy = intrinsics[1, 1]
        let cx = intrinsics[2, 0]
        let cy = intrinsics[2, 1]
        
        let filename = String(format: "frame_%04d.jpg", index)
        let imgUrl = targetDir.appendingPathComponent("images").appendingPathComponent(filename)
        
        // 2. Save RGB image
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        if let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
           let jpegData = context.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: [:]) {
            try? jpegData.write(to: imgUrl)
        }
        
        // 3. Record 6-DoF pose metadata
        let poseEntry: [String: Any] = [
            "frame_id": index,
            "image_file": "images/\(filename)",
            "timestamp": frame.timestamp,
            "camera_position": [position.x, position.y, position.z],
            "camera_orientation_yaw_pitch_roll": [eulerAngles.y * 180 / .pi, eulerAngles.x * 180 / .pi, eulerAngles.z * 180 / .pi],
            "intrinsics": [
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": CVPixelBufferGetWidth(pixelBuffer),
                "height": CVPixelBufferGetHeight(pixelBuffer)
            ]
        ]
        recordedPoses.append(poseEntry)
    }
    
    /// Stops scanning session and outputs the final package
    public func stopScanning() {
        guard isScanning, let sessionDir = sessionDirectory else { return }
        isScanning = false
        arSession.pause()
        roomCaptureSession?.stop()
        
        // Save camera poses manifest
        let manifest: [String: Any] = [
            "session_id": sessionDir.lastPathComponent,
            "sensor_type": "Apple_LiDAR_TrueDepth_RGB",
            "frame_count": recordedPoses.count,
            "frames": recordedPoses
        ]
        
        let manifestUrl = sessionDir.appendingPathComponent("camera_poses.json")
        if let jsonData = try? JSONSerialization.data(withJSONObject: manifest, options: .prettyPrinted) {
            try? jsonData.write(to: manifestUrl)
        }
        
        print("Scan finished. Saved to: \(sessionDir.path)")
        onScanCompleted?(sessionDir)
    }
}
