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
                pixelValues = image.toMLMultiArray(size: imgSize)
            }
            #elseif canImport(AppKit)
            if let image = NSImage(named: "test_case_0") {
                pixelValues = image.toMLMultiArray(size: imgSize)
            }
            #endif
        }
        
        // Fallback to dummy data
        inputIds = inputIds ?? createDummyInt32Array(shape: [1, NSNumber(value: seqLen)], fillValue: 1)
        attentionMask = attentionMask ?? createDummyInt32Array(shape: [1, NSNumber(value: seqLen)], fillValue: 1)
        bbox = bbox ?? createDummyInt32Array(shape: [1, NSNumber(value: seqLen), 4], fillValue: 0)
        tokenTypeIds = tokenTypeIds ?? createDummyInt32Array(shape: [1, NSNumber(value: seqLen)], fillValue: 0)
        
        // Always load an actual image file instead of a solid color placeholder for inference timing.
        // This ensures the hardware (especially Neural Engine) is evaluated against true visual variation/compression stats.
        if pixelValues == nil {
            #if canImport(UIKit)
            if let image = UIImage(named: "test_case_0") {
                pixelValues = image.toMLMultiArray(size: imgSize)
            } else {
                pixelValues = UIImage.solidColorImage(color: .gray, size: CGSize(width: imgSize, height: imgSize)).toMLMultiArray(size: imgSize)
            }
            #elseif canImport(AppKit)
            if let image = NSImage(named: "test_case_0") {
                pixelValues = image.toMLMultiArray(size: imgSize)
            } else {
                pixelValues = NSImage.solidColorImage(color: .gray, size: CGSize(width: CGFloat(imgSize), height: CGFloat(imgSize))).toMLMultiArray(size: imgSize)
            }
            #endif
        }

        guard let ids = inputIds, let mask = attentionMask, let box = bbox, let typeIds = tokenTypeIds, let pixels = pixelValues as? MLMultiArray else {
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
            } else {
                // If the generic feature name fails, try the generated property directly (if possible via Mirror)
                outputShapeStr = "Error: Output '\(outputName)' not found in prediction result."
            }
            
            return (outputShapeStr, timeElapsed)
        } catch {
            let errorMsg = "Inference error: \(error.localizedDescription)"
            print(errorMsg)
            return (errorMsg, 0)
        }
    }
}

#if canImport(UIKit)
extension UIImage {
    func toMLMultiArray(size: Int) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32) else { return nil }
        
        var pixels = [UInt8](repeating: 255, count: size * size * 4) // RGBA
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = size * 4
        
        guard let context = CGContext(data: &pixels,
                                      width: size,
                                      height: size,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return nil }
        
        // Fix orientation for UIImage drawing in CGContext
        context.translateBy(x: 0, y: CGFloat(size))
        context.scaleBy(x: 1.0, y: -1.0)
        
        // 1. Fill white background
        context.setFillColor(UIColor.white.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        
        // 2. Center and preserve aspect ratio
        UIGraphicsPushContext(context)
        let w = self.size.width
        let h = self.size.height
        let scale = CGFloat(size) / max(w, h)
        let nw = w * scale
        let nh = h * scale
        let targetRect = CGRect(x: (CGFloat(size) - nw) / 2.0, y: (CGFloat(size) - nh) / 2.0, width: nw, height: nh)
        self.draw(in: targetRect)
        UIGraphicsPopContext()
        
        // 3. Extract pixels directly into MLMultiArray with LayoutLMv3 Normalization
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: 1 * 3 * size * size)
        let mean: [Float32] = [0.485, 0.456, 0.406]
        let std: [Float32] = [0.229, 0.224, 0.225]
        
        for y in 0..<size {
            for x in 0..<size {
                let offset = (y * size + x) * 4
                // premultipliedLast -> R, G, B, A
                let r = Float32(pixels[offset]) / 255.0
                let g = Float32(pixels[offset + 1]) / 255.0
                let b = Float32(pixels[offset + 2]) / 255.0
                
                ptr[(0 * size + y) * size + x] = (r - mean[0]) / std[0]
                ptr[(1 * size + y) * size + x] = (g - mean[1]) / std[1]
                ptr[(2 * size + y) * size + x] = (b - mean[2]) / std[2]
            }
        }
        
        return array
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
    func toMLMultiArray(size: Int) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32) else { return nil }
        guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        
        var pixels = [UInt8](repeating: 255, count: size * size * 4) // RGBA
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = size * 4
        
        guard let context = CGContext(data: &pixels,
                                      width: size,
                                      height: size,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return nil }

        // 1. Fill white background
        context.setFillColor(NSColor.white.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))

        // 2. Draw image centered with aspect ratio preserved
        let w = self.size.width
        let h = self.size.height
        let scale = CGFloat(size) / max(w, h)
        let nw = w * scale
        let nh = h * scale
        let targetRect = CGRect(x: (CGFloat(size) - nw) / 2.0, y: (CGFloat(size) - nh) / 2.0, width: nw, height: nh)

        context.draw(cgImage, in: targetRect)
        
        // 3. Extract pixels and perform ImageNet bounds normalization
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: 1 * 3 * size * size)
        let mean: [Float32] = [0.485, 0.456, 0.406]
        let std: [Float32] = [0.229, 0.224, 0.225]
        
        for y in 0..<size {
            for x in 0..<size {
                let offset = (y * size + x) * 4
                // premultipliedLast -> R, G, B, A
                let r = Float32(pixels[offset]) / 255.0
                let g = Float32(pixels[offset + 1]) / 255.0
                let b = Float32(pixels[offset + 2]) / 255.0
                
                ptr[(0 * size + y) * size + x] = (r - mean[0]) / std[0]
                ptr[(1 * size + y) * size + x] = (g - mean[1]) / std[1]
                ptr[(2 * size + y) * size + x] = (b - mean[2]) / std[2]
            }
        }
        
        return array
    }
    
    static func solidColorImage(color: NSColor, size: CGSize) -> NSImage {
        let image = NSImage(size: size)
        image.lockFocus()
        let rect = NSRect(origin: .zero, size: size)
        color.set()
        rect.fill()
        image.unlockFocus()
        return image
    }
}
#endif
