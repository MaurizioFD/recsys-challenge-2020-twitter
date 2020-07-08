
# THIS SCRIPT ASSUMES THE
# DISK HAS A FILESYSTEM

mount_dir="/home/ubuntu/data"
disk_name="/dev/nvme0n1"

# show info about the disk
sudo file -s ${disk_name}

# directory where the disk will be mounted
sudo mkdir ${mount_dir}

# copy original fstab for backup
sudo cp /etc/fstab /etc/fstab.orig

# get disk size
disk_size=$(lsblk -no SIZE ${disk_name})

# check that disk size is 500 GB
if [ $disk_size == "500G" ]; then
    # get disk UUID
    uuid=$(lsblk -no UUID ${disk_name})  # should work like this, without 'sudo'
    echo "UUID for ${disk_name} is ${uuid}"

    # append UUID to fstab to automatically mount at boot
    sudo echo "UUID=${uuid}  ${mount_dir}  xfs  defaults,nofail  0  2" | sudo tee -a /etc/fstab

    # mount all
    sudo mount -a
else
    echo "Invalid disk size."
fi


echo "Done."