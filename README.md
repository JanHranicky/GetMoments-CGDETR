My fork of the [CG-DETR](https://github.com/wjun0830/CGDETR) repository. Only notable change is a pair of scripts for custom inference on test data, for more detail please go to the official repo.

# How to run GetMoments inference

1) Install needed libraries
```
pip install -r requirements.txt
```

2) Run inference script
```
python -m run_on_video.get_moments_run
```

It will do an inference on the test data saved in `run_on_video/example` and will dump the inference data in three separate pickle dictionaries. Each dictionary represensts different input query:

- No input query
- Input query consisting of constant 1 tensor
- Input query consisting of constant 0 tensor

3) Running the plotting script
 ```
python -m run_on_video.get_moments_plot
```

The plotting script will plot the saliance curve for all video clips, extract the 2s clip with maximum saliance score and extract the moment retrieved by the model. All outputs will be saved in their respective directories based on parsed predictions dictionary 

## ☑️ LICENSE
The annotation files and many parts of the implementations are borrowed from [Moment-DETR](https://github.com/jayleicn/moment_detr) and [QD-DETR](https://github.com/wjun0830/QD-DETR).
Our codes are under [MIT](https://opensource.org/licenses/MIT) license.
 
