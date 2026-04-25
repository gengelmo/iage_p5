# iage_p5

## Cómo ejecutar

1. Iniciar la máquina virtual de la práctica 4 (tiene que tener hadoop, los scripts en el $PATH, en entrono de python creado...).

2. Una vez iniciada, copiar el contenido de la práctica 5 en el $HOME de master
    Para ello, primero sacamos la ssh-config de master
    ```sh
    vagrant ssh-config > ssh-config
    ```
    Y después se copia la carpeta de la práctica 5 dentro.

    [!WARNING]
    Importante asegurar se de que la ruta a la carpeta de la práctica 5 está bien metida.

    ```sh
    scp -rF ssh_config RUTA_A_LA_CARPETA_DE_LA_P5 master:~/iage_prac_5
    ```

3. Una vez dentro, ejecutar [setup.sh](./setup.sh).
    ```sh
    chmod +x ~/iage_prac_5/setup.sh
    ~/iage_prac_5/setup.sh
    ```

4. Activar el entorno de python
    ```sh
    source ~/spark-env/bin/activate
    ```
5. Moverse a la carpeta de la práctica y ya se puede ejcutar [main.py](./main.py)
    ```sh
    cd iage_prac_5
    ```
    ```sh
    python main.py
    ```
