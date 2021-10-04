
cd /gpfs/data/rsingh47/masif/ChromAge/encode_histone_data/human

part1="/gpfs/data/rsingh47/masif/ChromAge/encode_histone_data/human/"
part2="raw_data/"

for d in */ ; do
cd /gpfs/data/rsingh47/masif/ChromAge/encode_histone_data/human/$d
for e in */ ; do
cd "$part1$d$e$part2"
for f in *.txt ; do
echo $f
wget -i $f
done
done
done
