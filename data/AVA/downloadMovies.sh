#!/bin/bash
# Download AVA original movies
set -e
terminal=`tty`
exec < ava_file_names_test_v2.1.txt

while read -r line
do
	{
		youtube-dl -f 'bestvideo' "http://www.youtube.com/watch?v=$line" -o "$line"
	} || {
		wget https://s3.amazonaws.com/ava-dataset/test/"$line"
	}

done
exec < "$terminal"
