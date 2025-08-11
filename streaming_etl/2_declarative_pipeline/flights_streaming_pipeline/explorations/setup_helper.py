class SetupHelper():
  @staticmethod
  def setup_schema(spark, catalog, db, reset_all_data, volume_name = None):
    if reset_all_data:
      print(f'clearing up volume named `{catalog}`.`{db}`.`{volume_name}`')
      try:
        spark.sql(f"DROP VOLUME IF EXISTS `{catalog}`.`{db}`.`{volume_name}`")
        spark.sql(f"DROP SCHEMA IF EXISTS `{catalog}`.`{db}` CASCADE")
      except Exception as e:
        print(f'catalog `{catalog}` or schema `{db}` do not exist.  Skipping data reset')

    def use_and_create_db(catalog, dbName, cloud_storage_path = None):
      print(f"USE CATALOG `{catalog}`")
      spark.sql(f"USE CATALOG `{catalog}`")
      spark.sql(f"""create database if not exists `{dbName}` """)

    assert catalog not in ['hive_metastore', 'spark_catalog'], "This demo only support Unity. Please change your catalog name."
    #If the catalog is defined, we force it to the given value and throw exception if not.
    current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
    if current_catalog != catalog:
      catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
      if catalog not in catalogs:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")

    use_and_create_db(catalog, db)

    print(f"using catalog.database `{catalog}`.`{db}`")
    spark.sql(f"""USE `{catalog}`.`{db}`""")    

    if volume_name:
      spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name};')
                     
  #Return true if the folder is empty or does not exists
  @staticmethod
  def is_folder_empty(folder):
    try:
      return len(dbutils.fs.ls(folder)) == 0
    except:
      return True
    
  @staticmethod
  def is_any_folder_empty(folders):
    return any([SetupHelper.is_folder_empty(f) for f in folders])

  @staticmethod
  def get_active_streams(start_with = ""):
    return [s for s in spark.streams.active if len(start_with) == 0 or (s.name is not None and s.name.startswith(start_with))]

  @staticmethod
  def stop_all_streams_async(start_with = "", sleep_time=0):
    import threading
    def stop_streams():
        SetupHelper.stop_all_streams(start_with=start_with, sleep_time=sleep_time)

    thread = threading.Thread(target=stop_streams)
    thread.start()

  @staticmethod
  def stop_all_streams(start_with = "", sleep_time=0):
    import time
    time.sleep(sleep_time)
    streams = SetupHelper.get_active_streams(start_with)
    if len(streams) > 0:
      print(f"Stopping {len(streams)} streams")
      for s in streams:
          try:
              s.stop()
          except:
              pass
      print(f"All stream stopped {'' if len(start_with) == 0 else f'(starting with: {start_with}.)'}")

  @staticmethod
  def wait_for_all_stream(start = ""):
    import time
    actives = SetupHelper.get_active_streams(start)
    if len(actives) > 0:
      print(f"{len(actives)} streams still active, waiting... ({[s.name for s in actives]})")
    while len(actives) > 0:
      spark.streams.awaitAnyTermination()
      time.sleep(1)
      actives = SetupHelper.get_active_streams(start)
    print("All streams completed.")