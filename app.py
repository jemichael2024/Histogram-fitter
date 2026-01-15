import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(layout="wide", page_title="Histogram Fitter")
#NICE COLOR SCHEME
st.markdown("""
<style>
.tip {
  border-bottom: 1px dotted #999;
  cursor: help;
}
/* COOL HOVER FEATURE */
.tip:hover::after {
  content: attr(data-tip);
  position: absolute;
  background: #333;
  color: #fff;
  padding: 6px;
  border-radius: 6px;
  width: 220px;
  font-size: 0.75rem;
  margin-left: 10px;
  z-index: 999;
  }
/* CHANGE BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #080019;   
}
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #443666;  
}

/* Change all text color in the entire app */
html, body, [class*="css"] {
    color: #ffffff !important; 
</style>
""", unsafe_allow_html=True)

# PAGE SETUP

st.title("Histogram Fitter")

#SIDEBAR: DATA INPUT
st.sidebar.header("DATA INPUT")

# Choose how to load data
input_mode = st.sidebar.radio(
    "Choose input method:",
    ["Enter Data (Table)", "Upload CSV"]
)

data = np.array([])

# OPTION 1: ENTER DATA BY TABLE
if input_mode == "Enter Data (Table)":
    st.sidebar.write("Enter your numbers:")

    # Empty table with 10 blank rows
    df = pd.DataFrame({"Values": [np.nan] * 10})

    edited = st.sidebar.data_editor(df, num_rows="dynamic")

    # Pull numbers and remove blanks
    data = pd.to_numeric(edited["Values"], errors="coerce").dropna().values


# OPTION 2: UPLOAD CSV FILE
else:
    uploaded = st.sidebar.file_uploader("Upload CSV File", type="csv")

    if uploaded is not None:

        # Read the raw text safely (NO pandas parsing)
        uploaded.seek(0)
        raw = uploaded.read().decode("utf-8", errors="ignore")

        numbers = []

        # Split the file into lines
        for line in raw.splitlines():

            # Replace commas with spaces so we can split cleanly
            line = line.replace(",", " ")

            # Split into values
            parts = line.split()

            # Convert each piece into a number if possible
            for p in parts:
                try:
                    numbers.append(float(p))
                except:
                    pass  # skip non-numeric text

        data = np.array(numbers)

        if len(data) > 0:
            st.sidebar.success(f"Loaded {len(data)} data points!")
        else:
            st.sidebar.error("No numeric values found in file!")
            
# Show status
if len(data) == 0:
    st.warning("No valid data yet. Enter values or upload a CSV.")
    st.stop()

with st.sidebar:
    st.badge("Success DATA added", icon=":material/check:", color="green")
st.success(f"Loaded {len(data)} values!")

# DISTRIBUTION SELECTION
left, right = st.columns([3, 4])

with left:
    st.header("Choose a Distribution")
    
    dist_names = ["uniform","expon","norm","triang",
                  "gamma", "weibull_min", "lognorm",
                  "weibull_min","gamma","beta","t"]
    dist_name = st.selectbox("Select Distribution: Odered from least to most complex", dist_names)
    dist = getattr(stats, dist_name)
    
    # Number of bins for histogram
    bins = st.slider("Number of Bins: Number of bars in a histogram", 1, 100, 30)
    
    # FIT THE DISTRIBUTION
    # Fit automatically using scipy
    params = dist.fit(data)
    
    # Extract loc & scale
    loc = params[-2]
    scale = params[-1]
    
    # Extract shape parameters 
    shape_params = params[:-2]
    
    # MANUAL FITTING MODE
    st.header("Manual Fitting (Optional)")
    manual_mode = st.checkbox("Adjust parameters manually")
    
    if manual_mode:
        st.write("Use these sliders to adjust the fitted curve manually.")
        #SHAPE PARAMETERS
        manual_shapes = []
        if len(shape_params) > 0:
            st.subheader("Shape Parameters")
            st.caption("These control the *shape* of the distribution (skew, tail weight, etc.)")
    
            for i, sp in enumerate(shape_params):
                manual_shapes.append(
                    st.slider(
                        f"Shape parameter {i+1}",
                        max(0.01, sp * 0.25),   # avoid invalid values
                        sp * 4,                 # avoid explosions
                        sp,                     # starting value
                        0.1
                    )
                )
        else:
            manual_shapes = []
            st.info("This distribution has **no shape parameters**.")
    
        #LOC PARAMETER
        st.subheader("Location (Shift)")
        st.caption("Shifts the entire curve left/right.")
        loc = st.slider(
            "Loc",
            data.min() - 10,
            data.max() + 10,
            loc,
            0.1
        )
        # SCALE PARAMETER
        st.subheader("Scale (Spread)")
        st.caption("Controls width/spread of the curve. Must be positive.")
    
        scale = st.slider(
            "Scale",
            max(0.01, scale * 0.25),   # minimum safe scale
            scale * 4,                 # max safe spread
            scale,
            0.1
        )
    else:
        # No manual override: use automatic fit
        manual_shapes = list(shape_params)
    
    # MAKe THE PLOT
    x = np.linspace(min(data), max(data), 300)
    
    if len(manual_shapes) > 0:
        pdf_y = dist.pdf(x, *manual_shapes, loc=loc, scale=scale)
    else:
        pdf_y = dist.pdf(x, loc=loc, scale=scale)
    
with right:    
    # PLOT
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(data, bins = bins, density = True, alpha=0.4, color = "red", label="Histogram")
    ax.plot(x, pdf_y, lw=2, color = "purple", label=f"{dist_name} PDF")
    ax.legend()
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    
    st.pyplot(fig)
    # Compare histogram heights with PDF values
    
    hist_y, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Get PDF values at histogram bin centers
    try:
        if len(manual_shapes) > 0:
            pdf_vals = dist.pdf(bin_centers, *manual_shapes, loc=loc, scale=scale)
        else:
            pdf_vals = dist.pdf(bin_centers, loc=loc, scale=scale)
    
        # Absolute errors
        errors = np.abs(hist_y - pdf_vals)
    
        avg_error = np.mean(errors)
        max_error = np.max(errors)
   
    except Exception as e:
        st.write("Error calculating fit quality:", e)
    #FIT ERROR CALCULATIONS
    errors = np.abs(hist_y - pdf_vals)
    
    # Basic error metrics
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Mean Squared Error
    mse = np.mean((hist_y - pdf_vals)**2)
    
    # R² Score
    ss_res = np.sum((hist_y - pdf_vals)**2)
    ss_tot = np.sum((hist_y - np.mean(hist_y))**2)
    r2 = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
    
    # Simple fit quality rating
    if r2 > 0.9:
        fit_quality = "Excellent"
    elif r2 > 0.75:
        fit_quality = "Good"
    elif r2 > 0.5:
        fit_quality = "Fair"
    else:
        fit_quality = "Poor"   
    # SHOW RESULTS
    side1, side2 = st.columns([3, 4])
    with side1:
        st.subheader("Fit Quality Results ")
        
        st.markdown('<span class="tip" data-tip="Average difference between your data and the fitted curve. Lower is better."> ℹ️ Average Error:</span>', unsafe_allow_html=True)
        st.write(f"{avg_error:.4f}")
        
        st.markdown('<span class="tip" data-tip="Largest single difference between your data and the curve.">ℹ️ Maximum Error:</span>', unsafe_allow_html=True)
        st.write(f"{max_error:.4f}")
        
        st.markdown('<span class="tip" data-tip="Mean Squared Error. Shows how far predictions are from true values.">ℹ️ MSE (Mean Squared Error):</span>', unsafe_allow_html=True)
        st.write(f"{mse:.4f}")
        
        st.markdown('<span class="tip" data-tip="R² measures how well the model explains variance in the data. 1 = perfect fit.">ℹ️ R² Score:</span>', unsafe_allow_html=True)
        st.write(f"{r2:.4f}")
        
        
    with side2:
        st.markdown('<span class="tip" data-tip="Name of the best-fitting distribution.">ℹ️ Distribution:</span>', unsafe_allow_html=True)
        st.write(dist_name)
        st.markdown('<span class="tip" data-tip="Rating based on how well the curve fits the data."> ℹ️ Fit Quality Rating:</span>', unsafe_allow_html=True)
        st.write(fit_quality)
        st.write("**Parameters:**", params)
        st.write("These parameters describe the shape and position of the distribution:")
        st.write("– The first value(s) control the shape, (pending on type)")
        st.write("– The second-to-last value shifts the curve left/right")
        st.write("– The last value controls how wide or spread-out the curve is")
  





