
## Setting OpenPCDet

- Execute the container
```
docker exec -it centerpointpillar bash
```

- Install OpenPCDet based CenterPointPillar
``` shell
cd ~/CenterPointPillar
sudo python setup.py develop
```

- To Build Python module, you have to install and wrap the c++ to python API.
``` shell
cd ~/
git clone https://github.com/pybind/pybind11.git
cd pybind11
cmake .
sudo make install

pip install --upgrade pip
sudo apt install python3-testresources

cd ~/CenterPointPillar/centerpoint/pybind
cmake -BRelease
cmake --build Release
```

## [Return to the main page.](../README.md)
