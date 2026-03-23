import CoreML
import Foundation

#if canImport(UIKit)
import UIKit
typealias PlatformImage = UIImage
#elseif canImport(AppKit)
import AppKit
typealias PlatformImage = NSImage
#endif

import CoreGraphics

struct TestCase: Codable {
    let input_ids: [Int]
    let attention_mask: [Int]
    let bbox: [[Int]]
    let token_type_ids: [Int]
    let image_size: [Int]
    let words: [String]
}

class CoreMLInference {
    // 1. Model instance
    let model: layoutlmv3_sparse
    
    // seq_len must match the fixed export shape
    let seqLen = 512
    let imgSize = 384
    
    init?() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            self.model = try layoutlmv3_sparse(configuration: config)
            print("Model loaded successfully.")
        } catch {
            print("Failed to load model: \(error)")
            return nil
        }
    }
    
    private func createInt32Array(from list: [Int], shape: [NSNumber]) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: shape, dataType: .int32) else { return nil }
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for (i, val) in list.enumerated() {
            if i < array.count {
                ptr[i] = Int32(val)
            }
        }
        return array
    }

    private func createBBoxArray(from list: [[Int]], shape: [NSNumber]) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: shape, dataType: .int32) else { return nil }
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for (i, box) in list.enumerated() {
            if i < shape[1].intValue {
                for (j, val) in box.enumerated() {
                    if j < 4 {
                        ptr[i * 4 + j] = Int32(val)
                    }
                }
            }
        }
        return array
    }
    
    private func createDummyInt32Array(shape: [NSNumber], fillValue: Int32 = 0) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: shape, dataType: .int32) else { return nil }
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for i in 0..<array.count { ptr[i] = fillValue }
        return array
    }

    func runInference(useRealData: Bool = false) -> (String, TimeInterval)? {
        var inputIds: MLMultiArray?
        var attentionMask: MLMultiArray?
        var bbox: MLMultiArray?
        var tokenTypeIds: MLMultiArray?
        var pixelValues: Any? // Can be MLMultiArray or CVPixelBuffer

        if useRealData, let jsonPath = Bundle.main.path(forResource: "test_case_0", ofType: "json"),
           let data = try? Data(contentsOf: URL(fileURLWithPath: jsonPath)),
           let testCase = try? JSONDecoder().decode(TestCase.self, from: data) {
            
            inputIds = createInt32Array(from: testCase.input_ids, shape: [1, NSNumber(value: seqLen)])
            attentionMask = createInt32Array(from: testCase.attention_mask, shape: [1, NSNumber(value: seqLen)])
            bbox = createBBoxArray(from: testCase.bbox, shape: [1, NSNumber(value: seqLen), 4])
            tokenTypeIds = createInt32Array(from: testCase.token_type_ids, shape: [1, NSNumber(value: seqLen)])
            
            #if canImport(UIKit)
            if let image = UIImage(named: "test_case_0") {
                pixelValues = image.toCVPixelBuffer(size: imgSize)
            }
            #elseif canImport(AppKit)
            if let image = NSImage(named: "test_case_0") {
                pixelValues = image.toCVPixelBuffer(size: imgSize)
            }
            #endif
        }
        
        // Fallback to dummy data
        inputIds = inputIds ?? createDummyInt32Array(shape: [1, NSNumber(value: seqLen)], fillValue: 1)
        attentionMask = attentionMask ?? createDummyInt32Array(shape: [1, NSNumber(value: seqLen)], fillValue: 1)
        bbox = bbox ?? createDummyInt32Array(shape: [1, NSNumber(value: seqLen), 4], fillValue: 0)
        tokenTypeIds = tokenTypeIds ?? createDummyInt32Array(shape: [1, NSNumber(value: seqLen)], fillValue: 0)
        
        // pixelValues dummy if still nil (assuming ImageType mode for now)
        if pixelValues == nil {
            #if canImport(UIKit)
            pixelValues = UIImage.solidColorImage(color: .gray, size: CGSize(width: imgSize, height: imgSize)).toCVPixelBuffer(size: imgSize)
            #elseif canImport(AppKit)
            pixelValues = NSImage.solidColorImage(color: .gray, size: CGSize(width: CGFloat(imgSize), height: CGFloat(imgSize))).toCVPixelBuffer(size: imgSize)
            #endif
        }

        guard let ids = inputIds, let mask = attentionMask, let box = bbox, let typeIds = tokenTypeIds, let pixels = pixelValues as? CVPixelBuffer else {
            return nil
        }

        do {
            let start = Date()
            let output = try model.prediction(
                input_ids: ids,
                attention_mask: mask,
                bbox: box,
                pixel_values: pixels,
                token_type_ids: typeIds
            )
            
            let timeElapsed = Date().timeIntervalSince(start)
            let outputName = "var_1295"
            var outputShapeStr = "Unknown"
            
            if let multiArray = output.featureValue(for: outputName)?.multiArrayValue {
                outputShapeStr = "\(outputName): \(multiArray.shape.map { $0.intValue })"
            }
            
            return (outputShapeStr, timeElapsed)
        } catch {
            print("Inference error: \(error)")
            return ("Error", 0)
        }
    }
}

#if canImport(UIKit)
extension UIImage {
    func toCVPixelBuffer(size: Int) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard status == kCVReturnSuccess else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: size, height: size, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.translateBy(x: 0, y: CGFloat(size))
        context?.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context!)
        draw(in: CGRect(x: 0, y: 0, width: size, height: size))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
    }
    
    static func solidColorImage(color: UIColor, size: CGSize) -> UIImage {
        let rect = CGRect(origin: .zero, size: size)
        UIGraphicsBeginImageContext(rect.size)
        let context = UIGraphicsGetCurrentContext()
        context!.setFillColor(color.cgColor)
        context!.fill(rect)
        let img = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return img!
    }
}
#elseif canImport(AppKit)
extension NSImage {
    func toCVPixelBuffer(size: Int) -> CVPixelBuffer? {
        guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard status == kCVReturnSuccess else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: size, height: size, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
    }
    
    static func solidColorImage(color: NSColor, size: CGSize) -> NSImage {
        let image = NSImage(size: size)
        image.lockFocus()
        color.drawSwap()
        let rect = NSRect(origin: .zero, size: size)
        color.set()
        rect.fill()
        image.unlockFocus()
        return image
    }
}
#endif
