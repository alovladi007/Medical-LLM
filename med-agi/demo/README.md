# Med-AGI System Demo

## 🌐 Live Demo Dashboard

A beautiful, interactive web dashboard showcasing the Med-AGI medical AI gateway system capabilities.

## 🚀 Quick Start

### Option 1: Python Server (Recommended)
```bash
cd /workspace/med-agi/demo
python3 server.py
```
Then open: **http://localhost:8080**

### Option 2: Direct File Access
Simply open `index.html` in your browser:
```bash
cd /workspace/med-agi/demo
open index.html  # macOS
xdg-open index.html  # Linux
start index.html  # Windows
```

## 📊 Demo Features

### Interactive Service Dashboard
- **Real-time Status Monitoring**: See which services are online
- **Service Cards**: Detailed information about each microservice
- **Feature Badges**: Quick view of capabilities (GPU, DICOM, etc.)

### Live Demos (Simulated)
1. **Imaging Analysis**
   - Select modality (CXR, CT, MRI)
   - Toggle GPU acceleration
   - View inference results with predictions

2. **EKG Analysis**
   - Generate sample waveform data
   - Analyze rhythm and intervals
   - View classification probabilities

3. **Model Evaluation**
   - Select models and datasets
   - View performance metrics
   - See confusion matrices

## 🎨 Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Smooth Animations**: Professional UI transitions
- **Interactive Elements**: Click to explore different services
- **Real-time Updates**: Status indicators with pulse animations
- **Gradient Themes**: Beautiful color schemes

## 📱 Screenshots

### Main Dashboard
- System status bar showing all services
- Service cards with detailed features
- Interactive demo buttons

### Demo Section
- Tabbed interface for different services
- Input controls for parameters
- Results display with JSON formatting
- Metrics visualization

## 🔧 Technical Details

- **Pure HTML/CSS/JavaScript**: No framework dependencies
- **Responsive Grid Layout**: Adapts to screen size
- **CSS Animations**: Smooth transitions and effects
- **Simulated API Responses**: Demo mode with realistic data

## 🌟 Service Status

| Service | Port | Status | Implementation |
|---------|------|--------|----------------|
| Imaging | 8006 | 🟢 Online | ✅ Complete |
| EKG | 8016 | 🟢 Online | ✅ Complete |
| Evaluation | 8005 | 🟢 Online | ✅ Complete |
| Anchor | 8007 | 🟡 Pending | 🔄 Ready |
| ModelCards | 8008 | 🟡 Pending | 🔄 Ready |
| Operations | 8010 | 🟡 Pending | 🔄 Ready |

## 🎯 Demo Scenarios

### Medical Imaging Analysis
1. Click "Try Demo" on Imaging Service card
2. Select imaging modality
3. Choose GPU or CPU processing
4. Click "Analyze Image"
5. View predictions and confidence scores

### EKG Rhythm Analysis
1. Click "Try Demo" on EKG Service card
2. Click "Generate Sample" for test data
3. Select lead configuration
4. Click "Analyze EKG"
5. View rhythm classification and intervals

### Model Performance Evaluation
1. Click "Try Demo" on Evaluation Service card
2. Enter model name
3. Select dataset
4. Click "Evaluate Model"
5. View metrics and confusion matrix

## 🛠️ Customization

To modify the demo:
1. Edit `index.html` for layout changes
2. Modify CSS styles in the `<style>` section
3. Update JavaScript functions for behavior
4. Change simulated data in the demo functions

## 📝 Notes

- This is a **demo interface** with simulated results
- Actual API integration requires running the full Med-AGI system
- Services marked "Coming Soon" are pending implementation
- All data shown is for demonstration purposes only

## 🔗 Related

- [Main README](/workspace/med-agi/README.md)
- [Services Documentation](/workspace/med-agi/SERVICES_IMPLEMENTATION.md)
- [API Documentation](/workspace/med-agi/docs/)

---

**Demo Version**: 1.0.0  
**Last Updated**: January 2025  
**Status**: 🟢 Running on http://localhost:8080