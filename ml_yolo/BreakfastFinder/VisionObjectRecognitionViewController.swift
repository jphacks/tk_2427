import UIKit
import AVFoundation
import Vision

class VisionObjectRecognitionViewController: ViewController {
    
    private var detectionOverlay: CALayer! = nil
    private var speechSynthesizer = AVSpeechSynthesizer() // 音声合成器のインスタンス
    private var trackedClass: String? // 追跡するオブジェクトのクラス

    // Vision parts
    private var requests = [VNRequest]()
    
    // 画面の幅を取得
    private var screenWidth: CGFloat {
        return UIScreen.main.bounds.width
    }
    
    // Visionの設定
    @discardableResult
    func setupVision() -> NSError? {
        // Setup Vision parts
        let error: NSError! = nil
        
        guard let modelURL = Bundle.main.url(forResource: "ObjectDetector", withExtension: "mlmodelc") else {
            return NSError(domain: "VisionObjectRecognitionViewController", code: -1, userInfo: [NSLocalizedDescriptionKey: "モデルファイルがありません"])
        }
        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            let objectRecognition = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                DispatchQueue.main.async(execute: {
                    // perform all the UI updates on the main queue
                    if let results = request.results {
                        self.drawVisionRequestResults(results)
                    }
                })
            })
            self.requests = [objectRecognition]
        } catch let error as NSError {
            print("モデルの読み込み中にエラーが発生しました: \(error)")
        }
        
        return error
    }
    
    func drawVisionRequestResults(_ results: [Any]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionOverlay.sublayers = nil // 古い認識されたオブジェクトをすべて削除
        
        var foundObject = false // オブジェクトが見つかったかどうかを追跡
        
        for observation in results where observation is VNRecognizedObjectObservation {
            guard let objectObservation = observation as? VNRecognizedObjectObservation else {
                continue
            }
            
            // 最も信頼度の高いラベルのみを選択
            let topLabelObservation = objectObservation.labels[0]
            let objectBounds = VNImageRectForNormalizedRect(objectObservation.boundingBox, Int(bufferSize.width), Int(bufferSize.height))
            
            // 追跡するオブジェクトのクラスがまだ決まっていない場合は、初回のオブジェクトを追跡
            if trackedClass == nil {
                trackedClass = topLabelObservation.identifier
            }
            
            // 追跡対象のクラスかどうかを確認
            if trackedClass == topLabelObservation.identifier {
                foundObject = true // オブジェクトが見つかった
                // オブジェクトの方向を判断
                let objectCenterX = objectBounds.midX
                let direction = objectCenterX < (screenWidth / 2) ? "左" : "右" // 中央より左か右かを判断
                
                // 音声で方向を読み上げる
                speakDirection(direction)
                
                let shapeLayer = self.createRoundedRectLayerWithBounds(objectBounds)
                
                let textLayer = self.createTextSubLayerInBounds(objectBounds,
                                                                identifier: topLabelObservation.identifier,
                                                                confidence: topLabelObservation.confidence)
                shapeLayer.addSublayer(textLayer)
                detectionOverlay.addSublayer(shapeLayer)
            }
        }
        
        // 追跡対象のオブジェクトが見つからなかった場合は、クラスをリセット
        if !foundObject {
            trackedClass = nil
        }
        
        self.updateLayerGeometry()
        CATransaction.commit()
    }
    
    // 指定した方向を音声で読み上げる
    func speakDirection(_ direction: String) {
        let utterance = AVSpeechUtterance(string: "\(direction)です")
        utterance.voice = AVSpeechSynthesisVoice(language: "ja-JP")
        speechSynthesizer.speak(utterance)
    }
    
    override func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let exifOrientation = exifOrientationFromDeviceOrientation()
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: exifOrientation, options: [:])
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
    
    override func setupAVCapture() {
        super.setupAVCapture()
        
        // Vision 部品の設定
        setupLayers()
        updateLayerGeometry()
        setupVision()
        
        // 二本指タップのジェスチャーを追加
        let doubleTapGesture = UITapGestureRecognizer(target: self, action: #selector(resetTracking))
        doubleTapGesture.numberOfTapsRequired = 2 // 二本タップ
        doubleTapGesture.numberOfTouchesRequired = 2 // 二本指でのタップ
        self.view.addGestureRecognizer(doubleTapGesture)
        
        // キャプチャを開始
        startCaptureSession()
    }
    
    @objc func resetTracking() {
        trackedClass = nil // 追跡対象をリセット
        detectionOverlay.sublayers?.removeAll() // レイヤーをリセット
        speechSynthesizer.stopSpeaking(at: .immediate) // 音声を停止
        print("カメラモードを初期状態に戻しました")
    }
    
    func setupLayers() {
        detectionOverlay = CALayer() // 観測のレンダリングをすべて含むコンテナレイヤー
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: bufferSize.width,
                                         height: bufferSize.height)
        detectionOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        rootLayer.addSublayer(detectionOverlay)
    }
    
    func updateLayerGeometry() {
        let bounds = rootLayer.bounds
        var scale: CGFloat
        
        let xScale: CGFloat = bounds.size.width / bufferSize.height
        let yScale: CGFloat = bounds.size.height / bufferSize.width
        
        scale = fmax(xScale, yScale)
        if scale.isInfinite {
            scale = 1.0
        }
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        
        // レイヤーを画面の向きに回転させ、スケーリングとミラーリングを行う
        detectionOverlay.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: scale, y: -scale))
        // レイヤーを中心に配置
        detectionOverlay.position = CGPoint(x: bounds.midX, y: bounds.midY)
        
        CATransaction.commit()
    }
    
    func createTextSubLayerInBounds(_ bounds: CGRect, identifier: String, confidence: VNConfidence) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        let formattedString = NSMutableAttributedString(string: String(format: "\(identifier)\n信頼度:  %.2f", confidence))
        let largeFont = UIFont(name: "Helvetica", size: 24.0)!
        formattedString.addAttributes([NSAttributedString.Key.font: largeFont], range: NSRange(location: 0, length: identifier.count))
        textLayer.string = formattedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.height - 10, height: bounds.size.width - 10)
        textLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        textLayer.shadowOpacity = 0.7
        textLayer.shadowOffset = CGSize(width: 2, height: 2)
        textLayer.foregroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])
        textLayer.contentsScale = 2.0 // Retina 描画
        // レイヤーを画面の向きに回転させ、スケーリングとミラーリングを行う
        textLayer.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: 1.0, y: -1.0))
        return textLayer
    }
    
    func createRoundedRectLayerWithBounds(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.backgroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1.0, 1.0, 0.2, 0.4])
        shapeLayer.cornerRadius = 7
        return shapeLayer
    }
}
