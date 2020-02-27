## IPL Human36MToolkit

#### Introduction

This is a toolkit for Human36m dataset.

Support:
- a pytorch custom dataset
    - loading data from Human36m dataset
    - data augmentation
        - flip
        - brightness
- 2D skeleton visualization

Before using this repository, you need to download the Human3.6m dataset from [google drive](https://drive.google.com/drive/folders/1kgVH-GugrLoc9XyvP6nRoaFpw3TmM5xK?usp=sharing).

Then extract all the data and make sure the data follow this structure:
```
Human3.6m
    - images
        - s_01_act_02_subact_01_ca_01
        - ...
    - annotations
        - Human36M_subject1_camera.json
        - ...
```

Finally, remove an empty folder `s_11_act_02_subact_02_ca_01` in `images`.

#### Usage

```shell script
cd {Path}
git clone https://github.com/ipl-uw/Human36MToolkit.git
ln -s {Path}/{This_Repository} {Path_to_Your_Code}
```

or

Clone this repo in the directory of your code
```shell script
cd {Path_to_Your_Code}
git clone https://github.com/ipl-uw/Human36MToolkit.git
```

In your code:

```python
import This_Repository
```
