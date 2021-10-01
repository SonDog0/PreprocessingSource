def set_logger(api, endpoint , save_path):
    mylogger = logging.getLogger("my")
    mylogger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)

    file_handler = logging.FileHandler('{}\{}_{}_{}.log'.format(save_path, api,endpoint,today_datetime), encoding='utf-8')
    mylogger.addHandler(file_handler)

    return mylogger
