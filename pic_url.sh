#!/bin/bash
echo "please input md_file and we will output the convert img_url md_file"
echo "$2 --> $3"
github_url='https://raw.githubusercontent.com/ManWingloeng/pic_store/master/'
jsdelivr_url='https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master/'
sed -e "s#$1/#${jsdelivr_url}/$1/#g" $2> $3
