#!/bin/bash
echo "please input md_file and we will output the convert img_url md_file"
echo "$1 --> $2"
github_url='https://raw.githubusercontent.com/ManWingloeng/pic_store/master/'
jsdelivr_url='https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master/'
sed -e "s#CVPR2020.assets/#${jsdelivr_url}/CVPR2020.assets/#g" $1> $2
