find_master_rev(){
  tempdir=$(mktemp -d)
  git clone $1 $tempdir/repo -q
  git -C $tempdir/repo rev-parse origin/master
  rm -rf $tempdir
}

RTRL_REV=${1:-$(find_master_rev git@github.com:rmst/rtrl.git)}
echo "Building for RTRL version $RTRL_REV"

docker build -t images.borgy.elementai.net/rmst/rtrl:$RTRL_REV --build-arg RTRL_REV=$RTRL_REV .
docker push images.borgy.elementai.net/rmst/rtrl:$RTRL_REV

# test with
# docker run -it rtrl bash