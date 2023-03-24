# Hyperspectral-Object-Tracking-SiamBAG

This is the source code for this paper called "SiamBAG: Band Attention Grouping-based Siamese Object Tracking Network for Hyperspectral Videos"

**After this paper is published, this source code of the SiamBAG tracker will be available. Please look forward to it**



# Visualization Tracking Results

**Video tracking results in some scenarios:**

https://user-images.githubusercontent.com/45682966/225011464-7e907999-2c1b-4b24-8c02-2d961f0d2e7e.mp4

**Note that:** in this video, this dataset is worker scenario, where the red bounding box is SiamBAG tracker, the blue one is ground truth.


**GIF tracking results in some scenarios:**
<table><tr>
  <td><img src="basketball.gif" alt="basketball" width="200px" height="100px"></td>
  <td><img src="car3.gif" alt="car3" width="200px" height="100px"></td>
  <td><img src="coke.gif" alt="coke" width="200px" height="100px"></td>
  <td><img src="forest2.gif" alt="forest2" width="200px" height="100px"></td>
</tr>
<tr>
  <td><img src="paper.gif" alt="paper" width="200px" height="100px"></td>
  <td><img src="pedestrian2.gif" alt="pedestrian2" width="200px" height="100px"></td>
  <td><img src="playground.gif" alt="playground" width="200px" height="100px"></td>
</tr></table>


**Note that**: in these scenarios, the black bounding box is SiamBAG tracker, the blue one is ground truth, and the red one is BAENet tracker. 
# Prerequisites

Some important environments for SiamBAG. 
1. **Python 3.9**
2. **PyTorch 1.13.0**
3. **CUDA 11.6**
Please refer to 'SiamBAG-Installation_Environment.txt' file for more detailed.
