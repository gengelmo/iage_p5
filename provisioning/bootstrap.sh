#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

if [ "$#" -ne 1 ]; then
    echo "Sintaxis: $0 MASTER_HOSTNAME"
    exit -1
fi

MASTER_HOSTNAME=$1

if [ -b /dev/sdb ]; then
    DISK0=/dev/sdb
elif [ -b /dev/vdb ]; then
    DISK0=/dev/vdb
fi

if [ -b /dev/sdc ]; then
    DISK1=/dev/sdc
elif [ -b /dev/vdc ]; then
    DISK1=/dev/vdc
fi

if [ -b /dev/sdd ]; then
    DISK2=/dev/sdd
elif [ -b /dev/vdd ]; then
    DISK2=/dev/vdd
fi

# Format and mount disks to be used with Hadoop HDFS and Spark
if [ ! -d "/data/disk0" ]; then
    mkdir -p /data/disk0 >& /dev/null
    mkfs.ext4 -F $DISK0
    mount $DISK0 /data/disk0
    chmod 1777 /data/disk0
else
    if ! grep -Fq $DISK0 /proc/mounts ; then
	mount $DISK0 /data/disk0 >& /dev/null
	chmod 1777 /data/disk0
    fi
fi

if [ ! -d "/data/disk1" ]; then
    mkdir -p /data/disk1 >& /dev/null
    mkfs.ext4 -F $DISK1
    mount $DISK1 /data/disk1
    chmod 1777 /data/disk1
else
    if ! grep -Fq $DISK1 /proc/mounts ; then
	mount $DISK1 /data/disk1
	chmod 1777 /data/disk1
    fi
fi

if [ ! -d "/data/disk2" ]; then
    mkdir -p /data/disk2 >& /dev/null
    mkfs.ext4 -F $DISK2
    mount /$DISK2 /data/disk2
    chmod 1777 /data/disk2
else
    if ! grep -Fq $DISK2 /proc/mounts ; then
	mount $DISK2 /data/disk2
	chmod 1777 /data/disk2
    fi
fi

if [ ! -d "/data/disk0/hdfs" ]; then
    mkdir /data/disk0/hdfs
fi

if [ ! -d "/data/disk0/spark-tmp" ]; then
    mkdir /data/disk0/spark-tmp
fi

if [ ! -d "/data/disk1/hdfs" ]; then
    mkdir /data/disk1/hdfs
fi

if [ ! -d "/data/disk1/spark-tmp" ]; then
    mkdir /data/disk1/spark-tmp
fi

if [ ! -d "/data/disk2/hdfs" ]; then
    mkdir /data/disk2/hdfs
fi

if [ ! -d "/data/disk2/spark-tmp" ]; then
    mkdir /data/disk2/spark-tmp
fi

chmod 1777 /data/disk0/hdfs
chmod 1777 /data/disk1/hdfs
chmod 1777 /data/disk2/hdfs
chmod 1777 /data/disk0/spark-tmp
chmod 1777 /data/disk1/spark-tmp
chmod 1777 /data/disk2/spark-tmp

if ! grep -Fq $DISK0 /etc/fstab ; then
    echo -e "$DISK0        /data/disk0     ext4    defaults,relatime       0       0" >> /etc/fstab
fi

if ! grep -Fq $DISK1 /etc/fstab ; then
    echo -e "$DISK1        /data/disk1     ext4    defaults,relatime       0       0" >> /etc/fstab
fi

if ! grep -Fq $DISK2 /etc/fstab ; then
    echo -e "$DISK2        /data/disk2     ext4    defaults,relatime       0       0" >> /etc/fstab
fi

systemctl unmask systemd-timesyncd
systemctl enable systemd-timesyncd.service
systemctl restart systemd-timesyncd.service

# Install software
rm -rf /var/lib/apt/lists/*
apt-get clean
apt-get update
SOFTWARE="nano sshpass unzip python3-venv python-apt-common fdisk dnsutils dos2unix whois nfs-common openjdk-21-jdk systemd-timesyncd"
echo "==> Installing software packages..."
if ! apt-get install -y -qq $SOFTWARE > /tmp/apt.log 2>&1; then
    echo "Error when installing software, log:"
    cat /tmp/apt.log
    exit 1
fi
echo "==> done"


# .profile
if ! grep -q "PATH=/sbin:\$PATH" /home/vagrant/.profile; then
  echo 'export PATH=/sbin:$PATH' >> /home/vagrant/.profile
fi

# NFS and SSH keys setup
SSH_PUBLIC_KEY=/share/.id_rsa.pub
SSH_DIR=/home/vagrant/.ssh

if [ ! -d "/share" ]; then
    mkdir /share >& /dev/null
fi

if grep -Fq /share /etc/fstab ; then
    sed -i "/share/d" /etc/fstab
fi

if [ "$(hostname)" = "$MASTER_HOSTNAME" ]; then
    # Install NFS server
    echo "==> Installing and configuring NFS server..."
    if ! apt-get install -y -qq nfs-kernel-server >/tmp/apt.log 2>&1; then
    	echo "Error when installing software, log:"
    	cat /tmp/apt.log
    	exit 1
    fi
    echo "==> done"

    if [ ! -f $SSH_DIR/id_rsa.pub ]; then
	# Create ssh keys
	echo -e 'y\n' | sudo -u vagrant ssh-keygen -t rsa -f $SSH_DIR/id_rsa -q -N ''

	if [ ! -f $SSH_DIR/id_rsa.pub ]; then
		echo "SSH public key could not be created"
		exit -1
	fi
    fi

    if [ ! -f /etc/ssh/ssh_config.d/90-key-checking.conf ]; then
        cat > /etc/ssh/ssh_config.d/90-key-checking.conf << EOF
Host *
    StrictHostKeyChecking no
EOF
    fi

    chown vagrant:vagrant $SSH_DIR/id_rsa*
    cp $SSH_DIR/id_rsa.pub $SSH_PUBLIC_KEY

    # Configure NFS export
    chmod 1777 /share
    sed -i "/share/d" /etc/exports
    echo -e "/share        192.168.56.0/24(rw,sync,no_subtree_check)" >> /etc/exports
    exportfs -ra
else
    sed -i '/127.0.1.1.*-worker/d' /etc/hosts
    umount /share >& /dev/null && sleep 2
    if ! grep -Fq /share /etc/fstab ; then
        echo -e "$MASTER_HOSTNAME:/share        /share     nfs    auto,relatime,tcp       0       0" >> /etc/fstab
    fi
    echo "Mounting NFS export"
    sleep 2 && mount $MASTER_HOSTNAME:/share /share
fi

if [ ! -f $SSH_PUBLIC_KEY ]; then
	echo "SSH public key does not exist"
	exit -1
fi

sed -i '/127.0.1.1.*packer-amd64/d' /etc/hosts
sed -i '/127.0.2.1/d' /etc/hosts
sed -i "/master/d" $SSH_DIR/authorized_keys >& /dev/null
cat $SSH_PUBLIC_KEY >> $SSH_DIR/authorized_keys
chown vagrant:vagrant $SSH_DIR/authorized_keys
chmod 0600 $SSH_DIR/authorized_keys

# Keep a full project copy in master's home for direct execution.
if [ "$(hostname)" = "$MASTER_HOSTNAME" ]; then
    PROJECT_SRC="/vagrant"
    PROJECT_DST="/home/vagrant/iage_prac_5"

    if [ -d "$PROJECT_SRC" ]; then
        mkdir -p "$PROJECT_DST"
        cp -a "$PROJECT_SRC"/. "$PROJECT_DST"/
        rm -rf "$PROJECT_DST/.vagrant"
        chown -R vagrant:vagrant "$PROJECT_DST"
    fi
fi

# Python virtual environment for project (master only)
if [ "$(hostname)" = "$MASTER_HOSTNAME" ]; then
    PROJECT_DST="/home/vagrant/iage_prac_5"
    VENV_DIR="$PROJECT_DST/.venv"
    BOOTSTRAP_SCRIPT="/home/vagrant/.iage_prac_5_bootstrap_venv.sh"

    cat > "$BOOTSTRAP_SCRIPT" << 'EOF'
#!/bin/bash
set -e

PROJECT_DIR="/home/vagrant/iage_prac_5"
VENV_DIR="$PROJECT_DIR/.venv"
SENTINEL_FILE="$VENV_DIR/.iage_packages_installed"
PIP_TMP_BASE="/data/disk0/pip-tmp"
PIP_CACHE_BASE="/data/disk0/pip-cache"

if [ ! -d "$PROJECT_DIR" ]; then
    exit 0
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

mkdir -p "$PIP_TMP_BASE" "$PIP_CACHE_BASE"
export TMPDIR="$PIP_TMP_BASE"
export PIP_CACHE_DIR="$PIP_CACHE_BASE"

if [ ! -f "$SENTINEL_FILE" ] || ! "$VENV_DIR/bin/python" -c "import numpy, pandas, matplotlib" >/dev/null 2>&1; then
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/pip" install --no-cache-dir numpy pandas matplotlib
    touch "$SENTINEL_FILE"
fi

EOF

    chown vagrant:vagrant "$BOOTSTRAP_SCRIPT"
    chmod 0755 "$BOOTSTRAP_SCRIPT"

    sed -i '/# AUTO_ACTIVATE_IAGE_VENV START/,/# AUTO_ACTIVATE_IAGE_VENV END/d' /home/vagrant/.bashrc
    cat >> /home/vagrant/.bashrc << 'EOF'

# AUTO_ACTIVATE_IAGE_VENV START
if [ -n "$PS1" ] && [ -d /home/vagrant/iage_prac_5 ]; then
    export TMPDIR=/data/disk0/pip-tmp
    export PIP_CACHE_DIR=/data/disk0/pip-cache

    if [ ! -f /home/vagrant/iage_prac_5/.venv/.iage_packages_installed ]; then
        /home/vagrant/.iage_prac_5_bootstrap_venv.sh >/tmp/iage_prac_5_venv_bootstrap.log 2>&1 || true
    fi

    if [ -f /home/vagrant/iage_prac_5/.venv/bin/activate ]; then
        . /home/vagrant/iage_prac_5/.venv/bin/activate
    fi

    if command -v spark-submit >/dev/null 2>&1; then
        SPARK_BIN_PATH="$(command -v spark-submit)"
        SPARK_HOME_CANDIDATE="$(dirname "$(dirname "$(readlink -f "$SPARK_BIN_PATH")")")"
        if [ -d "$SPARK_HOME_CANDIDATE/python" ]; then
            export SPARK_HOME="$SPARK_HOME_CANDIDATE"
            PY4J_ZIP="$(ls "$SPARK_HOME/python/lib"/py4j-*.zip 2>/dev/null | head -n 1)"
            if [ -n "$PY4J_ZIP" ]; then
                export PYTHONPATH="$SPARK_HOME/python:$PY4J_ZIP:${PYTHONPATH:-}"
            else
                export PYTHONPATH="$SPARK_HOME/python:${PYTHONPATH:-}"
            fi
        fi
    fi
fi
# AUTO_ACTIVATE_IAGE_VENV END
EOF
fi

# # Start Spark services
# export SPARK_HOME=/opt/spark
# 
# if [ "$(hostname)" = "$MASTER_HOSTNAME" ]; then
#     echo "Starting Spark Master..."
#     $SPARK_HOME/sbin/start-master.sh
# else
#     echo "Starting Spark Worker..."
#     # Wait a bit for master to start
#     sleep 10
#     $SPARK_HOME/sbin/start-worker.sh spark://$MASTER_HOSTNAME:7077
# fi