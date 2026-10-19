from glob import glob

from setuptools import setup

package_name = 'crane_vision_pick'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pai-student',
    maintainer_email='noreply@example.com',
    description='Vision-based pick and place for CRANE+ V2 (PAI final project)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_downsampler = crane_vision_pick.image_downsampler:main',
            'collect_demos = crane_vision_pick.collect_demos:main',
            'evaluate = crane_vision_pick.evaluate:main',
            'sim_check = crane_vision_pick.sim_check:main',
        ],
    },
)
