#!/bin/bash
cd ~/Documents
for file in *.mov; do
  if [ -f "$file" ]; then
    output="${file%.*}.mp4"
    ffmpeg -i "$file" -vcodec h264 -crf 23 -preset medium -r 24 "$output"
    echo "Compressed $file to $output"
  fi
done