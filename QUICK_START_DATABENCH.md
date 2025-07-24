# 🚀 DataBench Quick Start Guide

Get your DataBench web interface running in 3 simple steps!

## 📋 Prerequisites

- Python 3.8+ 
- `databench/` folder in your project directory
- Internet connection (for HuggingFace datasets)

## ⚡ Quick Setup

### Option 1: Automatic Setup (Recommended)
```bash
# Run the startup script - it will guide you through everything
python start_databench.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r databench_requirements.txt
pip install -r databench/requirements.txt

# 2. Start the API server
python databench_api.py

# 3. Open your browser to http://localhost:5002/
```

## 🧪 Test Your Setup

```bash
# Test that everything is working
python test_databench_api.py
```

## 🎯 Using the Web Interface

1. **Open Browser**: Visit `http://localhost:5002/`

2. **Enter Dataset**: Type a HuggingFace dataset path like:
   - `gribok201/150episodes6`
   - `lerobot/pusht_image`

3. **Select Metrics**: Click on the metric cards you want:
   - **Action Consistency** - Visual-text alignment
   - **Visual Diversity** - Scene variation 
   - **High-Fidelity Vision** - Quality assessment
   - **Trajectory Quality** - Data completeness
   - **Dataset Coverage** - Scale and diversity
   - **Robot Action Quality** - Motion smoothness

4. **Configure Options**:
   - **Subset Size**: Start with 10-50 for quick testing
   - **HF Token**: Only needed for private datasets

5. **Run Evaluation**: Click "Start Evaluation" and wait for results!

## ⚡ Quick Examples

### Fast Test (Recommended for first try)
- Dataset: `gribok201/150episodes6`
- Metrics: Select just `a` (Action Consistency)
- Subset: `10`
- Expected time: 2-5 minutes

### Full Evaluation
- Dataset: `gribok201/150episodes6` 
- Metrics: Select all (`a,v,h,t,c,r`)
- Subset: `100`
- Expected time: 15-30 minutes

## 🔧 Troubleshooting

### "Failed to fetch" Error
```bash
# The API server isn't running. Start it:
python databench_api.py

# Then refresh your browser
```

### "Cannot connect to DataBench server"
```bash
# Check if server is running on port 5002:
curl http://localhost:5002/health

# If not, start it:
python databench_api.py
```

### Missing Dependencies
```bash
# Install Flask dependencies:
pip install flask flask-cors requests

# Install DataBench dependencies:
pip install -r databench/requirements.txt
```

### DataBench Script Not Found
```bash
# Verify the databench folder structure:
ls -la databench/scripts/evaluate.py

# Should exist. If not, check that you have the databench folder.
```

## 📊 Understanding Results

Results are color-coded:
- **🟢 Green (0.8-1.0)**: Excellent quality ⭐⭐⭐⭐⭐
- **🔵 Blue (0.6-0.8)**: Good quality ⭐⭐⭐⭐
- **🟡 Yellow (0.4-0.6)**: Moderate quality ⭐⭐⭐
- **🟠 Orange (0.2-0.4)**: Poor quality ⭐⭐
- **🔴 Red (0.0-0.2)**: Very poor quality ⭐

## 💡 Pro Tips

1. **Start Small**: Use subset=10 for your first test
2. **Select Specific Metrics**: Only run what you need
3. **Private Datasets**: Get your HF token from https://huggingface.co/settings/tokens
4. **Check Progress**: The evaluation shows real-time progress
5. **Save Results**: Results are automatically saved in the `results/` folder

## 🆘 Still Having Issues?

1. **Check the logs**: Look at the terminal running `databench_api.py`
2. **Test the CLI**: Try `python databench/scripts/evaluate.py --help`
3. **Verify setup**: Run `python test_databench_api.py`
4. **Check ports**: Make sure port 5002 isn't being used by another app

## 📈 Next Steps

Once you have DataBench working:
- Try different datasets from HuggingFace
- Experiment with different metric combinations
- Use the results to improve your robot training data
- Check out the full documentation in `README-DATABENCH.md`

---

🤖 **Happy Robot Dataset Evaluation!** 

For detailed documentation, see `README-DATABENCH.md` 