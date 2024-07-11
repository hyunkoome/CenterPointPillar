from pcdet.models.detector3d.centerpoint_pillar.centerpoint import CenterPoint

__all__ = {
    'CenterPoint': CenterPoint,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
