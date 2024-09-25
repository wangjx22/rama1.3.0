# Rama Scoring

docker image:
```commandline
harbor.biomap-int.com/ziqiao/folding_scoring:v1.0
```

inference:
Use k8s pipeline for inference.
pipeline name: 
```commandline
ziqiao-folding-scoring-nb-inference
```

params:
```commandline
input_path: input txt or csv containing paths of pdbs. see examples recorded in the model zoo.
ckpt_path: ckpt path
output_dir: dir for output txt. output txt name is inference.txt
yaml_path: config yaml path
working_dir: dir for the code work space.
```

