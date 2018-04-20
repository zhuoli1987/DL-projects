# DL-projects
This is a repository that aims at prototyping deep learning algorithms into deployable products.

# Installation
Use the [Anaconda](https://conda.io/docs/index.html) environment simply execute the following command.

Windows:```activate DL_projects```

maxOS/Linux:```source activate DL_projects```

# Projects
The following is a list of abstracts of all projects in this repository.

## Flask_DL
This project is about to using [Flask](http://flask.pocoo.org/) framework to build a deep learning backend service that can be accessed through RESTful APIs. The ultimate goal is to deploy the service onto cloud based platform. The service is the dog breed identifier that I built.

## IESO
> This project is about to use RNN to predict the Ontario hourly energy demand. 

All the data is coming from [IESO](http://www.ieso.ca/). The ultimate goal is to make this a product that can predict the peak demand and send out alarms to the users. This can be useful to the Ontario ClassA consumers who suffer from the [Global Adjustment](http://www.ieso.ca/power-data/price-overview/global-adjustment).

## Comic_GAN
> This project is about to use DCGAN to generate manga/comic character faces. 

### Data Set
Images are collected from [danbooru](http://danbooru.donmai.us/) using [icrawler](https://github.com/hellock/icrawler) and preprocessed using [python-animeface](https://github.com/nya3jp/python-animeface). 

Thanks to @jayleicn, an alternative data set that contains 115085 images in 126 tags can be found at 

- Brine (a python-based dataset management library): https://www.brine.io/jayleicn/anime-faces 
- Google Drive: https://drive.google.com/file/d/0B4wZXrs0DHMHMEl1ODVpMjRTWEk/view?usp=sharing
- BaiduYun: https://pan.baidu.com/s/1o8Nxllo

Non-commercial use please.

### Usage
- Download the data set and extract the images into a folder
- Usage the following command to start the jupyter notebook
```bash
# start the jupyter notebook server
jupyter notebook
```
- Open the 'Comic-GAN' notebook and run all.
