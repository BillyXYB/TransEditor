# perturbation ranges and layers for stylegan2, stylegan_idvert,
# and stylegan2_ada models, for each dataset domain

stylegan2_settings = {
    'ffhq': {
        'isotropic_eps_fine': [0.1, 0.2, 0.3],
        'isotropic_eps_coarse': [0.1, 0.2, 0.3],
        'pca_eps': [1.0, 2.0, 3.0],
        'pca_stats': 'networks/stats/stylegan2_ffhq_stats.npz',
        'fine_layer': 10,
        'coarse_layer': 4,
    },
    'car': {
        'isotropic_eps_fine': [0.3, 0.5, 0.7],
        'isotropic_eps_coarse': [1.0, 1.5, 2.0],
        'pca_eps': [1.0, 2.0, 3.0],
        'pca_stats': 'networks/stats/stylegan2_car_stats.npz',
        'fine_layer': 10,
        'coarse_layer': 4,
    },
    'cat': {
        'isotropic_eps_fine': [0.1, 0.2, 0.3],
        'isotropic_eps_coarse': [0.5, 0.7, 1.0],
        'pca_eps': [0.5, 0.7, 1.0],
        'pca_stats': 'networks/stats/stylegan2_cat_stats.npz',
        'fine_layer': 10,
        'coarse_layer': 4,
    },
}

stylegan_idinvert_settings = {
    'ffhq': {
        'coarse_layer': 4,
        'fine_layer': 10,
    },
}

stylegan2_cc_settings = {
    'cifar10': {
        'fine_layer': 7,
        'cc_mean_w': 'networks/stats/stylegan2_cifar10c_wmean.npz'
    },
}
