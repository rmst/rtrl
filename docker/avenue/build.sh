find_master_rev(){
  tempdir=$(mktemp -d)
  git clone $1 $tempdir/repo -q
  git -C $tempdir/repo rev-parse origin/master
  rm -rf $tempdir
}

AVENUE_REV=${1:-$(find_master_rev git@github.com:ElementAI/Avenue.git)}
echo "Building with Avenue revision $AVENUE_REV"

git clone git@github.com:ElementAI/Avenue.git /app/avenue
pushd avenue
  git pull
  git reset --hard ${AVENUE_REV}
  pip install -e .
popd

docker build -t unity3d - < unity3d.dockerfile
docker build -t unity3d-py --build-arg BASE=unity3d:latest --build-arg AVENUE_REV=$AVENUE_REV ../pytorch
docker build -t avenue --build-arg BASE=unity3d-py:latest . 
docker tag avenue:latest images.borgy.elementai.net/rmst/avenue:latest
docker push images.borgy.elementai.net/rmst/avenue:latest

rm -rf avenue