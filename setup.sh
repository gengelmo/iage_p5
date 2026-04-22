
# Iniciar el distributed file system
start-dfs.sh
# Iniciar los nodos
start-all.sh
# Crear la carpeta en el dfs
hdfs dfs -mkdir -p /user/vagrant/data/Stocks
# Copiar los datos a la carpeta
hdfs dfs -put /home/vagrant/iage_prac_5/data/Stocks/*.txt /user/vagrant/data/Stocks/
