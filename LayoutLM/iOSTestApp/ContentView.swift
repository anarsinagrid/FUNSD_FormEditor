import SwiftUI

struct ContentView: View {
    
    @State private var resultText = "Ready to test CoreML model"
    @State private var timeText = ""
    @State private var isRunning = false
    @State private var useRealData = false
    
    @State private var runner: CoreMLInference? = CoreMLInference()

    var body: some View {
        ScrollView {
            VStack(spacing: 25) {
                Image(systemName: "cpu")
                    .imageScale(.large)
                    .font(.system(size: 60))
                    .foregroundColor(.blue)
                    .padding(.top, 20)
                
                Text("LayoutLM CoreML Test")
                    .font(.title)
                    .bold()
                
                Picker("Test Mode", selection: $useRealData) {
                    Text("Dummy Data").tag(false)
                    Text("Real Document (#0)").tag(true)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)

                if useRealData {
                    VStack(spacing: 8) {
                        Text("Standard FUNSD Test Image:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Image("test_case_0")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(height: 200)
                            .cornerRadius(12)
                            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.gray.opacity(0.3), lineWidth: 1))
                            .help("FUNSD Test Document #0")
                    }
                }
                
                VStack(spacing: 12) {
                    Text(resultText)
                        .font(.system(.body, design: .monospaced))
                        .multilineTextAlignment(.center)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(12)
                    
                    if !timeText.isEmpty {
                        Text(timeText)
                            .font(.headline)
                            .foregroundColor(.green)
                    }
                }
                .padding(.horizontal)
                
                Button(action: {
                    runTest()
                }) {
                    HStack {
                        if isRunning {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Image(systemName: "play.fill")
                        }
                        Text(isRunning ? "Running..." : "Run Inference")
                            .bold()
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(isRunning ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(16)
                    .shadow(radius: 4)
                }
                .disabled(isRunning)
                .padding(.horizontal, 40)
                
                if runner == nil {
                    Text("Error: Failed to load LayoutLM model")
                        .foregroundColor(.red)
                        .font(.caption)
                } else {
                    Text("Model loaded — seq_len=512, input: Tensor (384x384)")
                        .foregroundColor(.green)
                        .font(.caption)
                }
                
                Spacer()
            }
            .padding()
        }
    }
    
    func runTest() {
        guard let runner = runner else {
            resultText = "Model not initialized"
            return
        }
        
        isRunning = true
        resultText = "Running inference..."
        timeText = ""
        
        DispatchQueue.global(qos: .userInitiated).async {
            if let (shape, time) = runner.runInference(useRealData: useRealData) {
                DispatchQueue.main.async {
                    self.resultText = "Success!\nOutput shape:\n\(shape)"
                    self.timeText = String(format: "Execution Time: %.2f ms", time * 1000)
                    self.isRunning = false
                }
            } else {
                DispatchQueue.main.async {
                    self.resultText = "Inference Failed.\nCheck console logs for data loading errors."
                    self.isRunning = false
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
