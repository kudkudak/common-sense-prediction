from src.utils.vegab import ConfigRegistry

config = ConfigRegistry()

config.set_root_config({
    'batch_size': 2.,
    'n_epochs': 2,
    'rel_init': 0.02,
    'rel_vec_size': 200,
    'activation': 'relu',
    'hidden_units': 1000,
    'lambda': 0.001,
    'learning_rate': 0.01,
})
