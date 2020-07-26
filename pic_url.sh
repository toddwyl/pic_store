#!/bin/bash
echo "please input md_file and we will output the convert img_url md_file"
echo "$1.md --> $1_url.md"
github_url='https://raw.githubusercontent.com/ManWingloeng/pic_store/master/'
jsdelivr_url='https://cdn.jsdelivr.net/gh/ManWingloeng/pic_store@master/'
sed -e "s#$1.assets/#${jsdelivr_url}/$1.assets/#g" $1.md> $1_url.md
