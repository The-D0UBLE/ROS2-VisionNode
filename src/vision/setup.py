from setuptools import find_packages, setup
from glob import glob

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

                # 👇 install models
        ('share/' + package_name + '/models',
         glob('vision/models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='araf',
    maintainer_email='araf@todo.todo',
    description='Vision node for ROS2',
    license='',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_node = vision.vision_node:main',
        ],
    },
)
