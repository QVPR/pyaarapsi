#!/bin/sh

# This script will grab files off google drive
# It will try to use curl to grab the output, but sometimes google drive likes
# to prank us and will send us to a html page instead. In this case, we need
# to grab out a confirm string from the html page, so we save that to our file
# system, grep out the string, then curl the real file with similar command.
# Bit silly, but works consistently and also doesn't need any python packages
# or authentication / log-in to google (which would invite a selection of
# other issues we do not want to deal with).
filename=$2
driveid=$1
# Try download the file:
curl -L "https://drive.usercontent.google.com/download?id=$driveid" -o $filename
# Check if file is correct:
filesize="$(stat -c%s $filename)"
minsize=3000
echo "Filesize: $filesize"

if [ "$filesize" -lt "$minsize" ]; then
	confirmvariable=$(cat $filename | grep -oP '(?<=confirm" value=").*(?="><input type="hidden")')
	#confirmvariable="$(curl -L "https://drive.usercontent.google.com/download?id=$driveid" | grep -oP '(?<=confirm" value=").*(?="><input type="hidden")')"
	curl -L "https://drive.usercontent.google.com/download?id=$driveid&confirm=$confirmvariable" -o $filename
fi
