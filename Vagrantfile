# -*- mode: ruby -*-
# vi: set ft=ruby :
require_relative 'provisioning/vbox.rb'
VBoxUtils.check_version('7.2.4')
Vagrant.require_version ">= 2.4.9"

# Modifica la variable STUDENT_PREFIX para sustituir "X" por tu prefijo
# Ejemplo, el alumno Roberto Rey Expósito, que hace la práctica en el curso
# 25/26, utilizará el siguiente prefijo: rre2526
STUDEN_PREFIX = "haa2526"

# Hostnames for master and worker nodes
MASTER_HOSTNAME = "#{STUDEN_PREFIX}-master"
WORKER_HOSTNAME = "#{STUDEN_PREFIX}-worker"

# Master settings
MASTER_IP = "192.168.56.10"
MASTER_CORES = 1
MASTER_MEM = 2048
MASTER_NUM_DISKS=3
MASTER_DISK_SIZE='4GB'
# Worker settings
NUM_WORKERS = 3
WORKER_CORES = 1
WORKER_MEM = 2048
WORKER_NUM_DISKS=3
WORKER_DISK_SIZE='8GB'

require 'ipaddr'
CLUSTER_IP_ADDR = IPAddr.new MASTER_IP
CLUSTER_IP_ADDR = CLUSTER_IP_ADDR.succ

ENV["VAGRANT_NO_PARALLEL"] = "yes"

Vagrant.configure("2") do |config|
  config.vm.box = "rreye/debian-13"
  config.vm.box_version = "20260304"
  config.vm.box_check_update = false
  config.vm.synced_folder ".", "/vagrant"

  # Configure hostmanager plugin
  config.hostmanager.enabled = true
  config.hostmanager.manage_host = true
  config.hostmanager.manage_guest = true

  # Master node
  config.vm.define "master", primary: true do |master|
    master.vm.hostname = MASTER_HOSTNAME
    master.vm.network "private_network", ip: MASTER_IP

    for i in 1..MASTER_NUM_DISKS do
        master.vm.disk :disk, size: MASTER_DISK_SIZE, primary: false, name: "disk#{i}"
    end
      
    master.vm.provider "virtualbox" do |vb|
        vb.name = "IAGE-#{master.vm.hostname}"
        vb.customize ["modifyvm", :id, "--graphicscontroller", "vmsvga"]
        vb.cpus = MASTER_CORES
        vb.memory = MASTER_MEM
        vb.gui = false
    end
    
    master.vm.provider "libvirt" do |libv|
        libv.title = "IAGE-#{master.vm.hostname}"
        libv.driver = "kvm"
        libv.cpus = MASTER_CORES
        libv.memory = MASTER_MEM
        libv.cputopology :sockets => '1', :cores => MASTER_CORES, :threads => '1'
        
        for i in 1..MASTER_NUM_DISKS do
            libv.storage :file, :size => MASTER_DISK_SIZE
        end
    end
  end
  
  # Worker nodes
  (1..NUM_WORKERS).each do |i|
    config.vm.define "worker#{i}" do |worker|
	      worker.vm.hostname = "#{WORKER_HOSTNAME}#{i}"
	      IP_ADDR = CLUSTER_IP_ADDR.to_s
        CLUSTER_IP_ADDR = CLUSTER_IP_ADDR.succ
        worker.vm.network "private_network", ip: IP_ADDR
        
        for i in 1..WORKER_NUM_DISKS do
            worker.vm.disk :disk, size: WORKER_DISK_SIZE, primary: false, name: "disk#{i}"
        end
    
        worker.vm.provider "virtualbox" do |vb|
	          vb.name = "IAGE-#{worker.vm.hostname}"
	          vb.customize ["modifyvm", :id, "--graphicscontroller", "vmsvga"]
            vb.cpus = WORKER_CORES
            vb.memory = WORKER_MEM
	          vb.gui = false
        end
        
        worker.vm.provider "libvirt" do |libv|
            libv.title = "IAGE-#{worker.vm.hostname}"
            libv.driver = "kvm"
            libv.cpus = WORKER_CORES
            libv.memory = WORKER_MEM
            libv.cputopology :sockets => '1', :cores => WORKER_CORES, :threads => '1'
            
            for i in 1..WORKER_NUM_DISKS do
                libv.storage :file, :size => WORKER_DISK_SIZE
            end
        end
    end
  end
  
  # Global provisioning bash script
  config.vm.provision "global", type: "shell", run: "always", path: "provisioning/bootstrap.sh" do |script|
      script.args = [MASTER_HOSTNAME]
  end
end
