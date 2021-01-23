# vid2vid--paddle

### About this project

This project is paddle implementation for few-shot photorealistic video-to-video translation. Go to [this link](https://arxiv.org/pdf/1910.12713) for more details. This project is heavily copied from original project [Few-shot vid2vid](https://github.com/NVlabs/few-shot-vid2vid) and [Imaginaire](https://github.com/NVlabs/imaginaire). This work is made available under the Nvidia Source Code License (1-Way Commercial). To view a copy of this license, visit [License](https://nvlabs.github.io/few-shot-vid2vid/License.txt). If this work is benifit for you, please cite:

@inproceedings{wang2018fewshotvid2vid,\
   author    = {Ting-Chun Wang and Ming-Yu Liu and Andrew Tao and Guilin Liu and Jan Kautz and Bryan Catanzaro},\
   title     = {Few-shot Video-to-Video Synthesis},\
   booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},   \
   year      = {2019},\
}


### Dependencies

This project is totally conducted in AI Studio, Because of the lack of ecology in [AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1422764), other dependencies should implemented first to support this project. Here I implemented 
* [openpose-paddle](https://aistudio.baidu.com/aistudio/projectdetail/1403743)
* [densepose-paddle](https://aistudio.baidu.com/aistudio/projectdetail/1413614)
* [LiteFlowNet-paddle](https://aistudio.baidu.com/aistudio/projectdetail/1354069)
* [preprocess](https://aistudio.baidu.com/aistudio/projectdetail/1441848)


### Dataset

**YouTube Dancing Videos Dataset**

YouTube Dancing Videos Dataset is a large scale dancing video dataset collected from *YouTube* site. Note that the dataset used in this project is slightly different from the original project with augmentation data collected from *bilibili*. In the end, 700 raw videos froms the raw video dataset. Then, I use [openpose-paddle](https://aistudio.baidu.com/aistudio/projectdetail/1403743) and [densepose-paddle](https://aistudio.baidu.com/aistudio/projectdetail/1413614) to extract pose annotations from the raw videos. Finially, I get 4240 video sequences, and 1,382,329 raw frames with corresponding pose annotations. This dataset is split into 4 subdataset for convenience of training and the constriction of storage. For more details, please go to [preprocess](https://aistudio.baidu.com/aistudio/projectdetail/1441848) and [YouTube Dancing Videos Dataset](https://aistudio.baidu.com/aistudio/projectdetail/1469795).

**For other dataset**

Not implemented here!

### Train

```
! cd /home/aistudio/vid2vid/ && python ./train.py --logdir /path/to/log/dictory/ \
                                   --max_epoch 20 \
                                   --max_iter_per_epoch 10000 \
                                   --num_epochs_temporal_step 4 \
                                   --train_data_root /path/to/dancing/video/dataset \
                                   --val_data_root /path/to/evaluation/dataset
```

To train your own dataset, please follow the instruction in [preprocess](https://aistudio.baidu.com/aistudio/projectdetail/1441848) to get your dataset first, then run command above. Be careful of **mode collapse** when training with highly inner-variance dataset.


### Pretrained Models

Here I trained four models for 4 subdataset described above. Note that these four models are train on YouTube Dancing Videos Dataset, you need to finetune on your own dataset to synthesis your own videos.

I put these pretrained models to [here](https://aistudio.baidu.com/aistudio/datasetdetail/68795). Model trained on set 4 have not be released currently.

### Evaluation

```
! cd /home/aistudio/vid2vid/ && python ./evaluate.py --log_dir /path/to/evaluation/results/output/directory
                                     --checkpoint_logdir /path/to/checkpoints/directory
                                     --eval_data_dir /path/to/eval/data/directory
```
For example, to evaluate model trained on set-1, I can run
```
python ./evaluate.py --logdir /home/aistudio/vid2vid/outputs/evaluation/ 
               --checkpoint_logdir /home/aistudio/work/logs/checkpoints/ 
               --eval_data_dir /home/aistudio/data/data68795/home/aistudio/test_pose/1_pose/images/
```


### Acknowledge

Thanks for the support of GPU resourses provided by **AI Studio** for this project.
