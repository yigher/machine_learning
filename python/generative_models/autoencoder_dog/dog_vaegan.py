import libs.vaegan as vae

vae.test_dog(
    n_epochs=5000,
    filter_sizes=[5, 5, 5],
    n_filters=[64, 64, 64],
    n_hidden=128,
    n_code=32,
    input_shape=[299, 299, 3],
    crop_shape=[64, 64, 3],
    log_file="logs/dog.log")
