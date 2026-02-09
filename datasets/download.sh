file=$1

while read line 
do
  filename=$(basename "$line")
  if find . -type f -name "$filename" | grep -q .; then
    echo "已下载: $filename"
  else
    wget "$line"
  fi
done <$file