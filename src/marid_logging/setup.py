from setuptools import find_packages, setup

package_name = 'marid_logging'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MARID',
    description='MARID UAV sensor data logging (IMU, etc.) to CSV for ML and analysis.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'imu_logger = marid_logging.imu_logger:main',
            'marid_data_logger = marid_logging.marid_data_logger:main',
            'pose_estimator_logger = marid_logging.pose_estimator_logger:main',
        ],
    },
)
