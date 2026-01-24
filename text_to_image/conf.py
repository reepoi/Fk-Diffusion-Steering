import logging
from dataclasses import field
from pathlib import Path
import sys
from typing import Any, List

import hydra
import hydra_orm.utils
import omegaconf
from omegaconf import OmegaConf
import sqlalchemy as sa
from hydra_orm import orm


DIR_ROOT = Path(__file__).parent


def filename_relative_to_dir_root(filename):
    return Path(filename).relative_to(DIR_ROOT)


def getLoggerByFilename(filename):
    return logging.getLogger(str(filename_relative_to_dir_root(filename)))


def get_engine(dir=str(DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


engine = get_engine()
orm.create_all(engine)
Session = sa.orm.sessionmaker(engine)


HYDRA_INIT = dict(version_base=None, config_path='conf', config_name='conf')


def get_run_dir(hydra_init=HYDRA_INIT, commit=True, engine_name='runs'):
    if '-m' in sys.argv or '--multirun' in sys.argv:
        raise ValueError("The flags '-m' and '--multirun' are not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = get_engine(name=engine_name)
        orm.create_all(engine)
        with sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            # if commit and '-c' not in sys.argv:
            if commit:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


def set_run_dir(last_override, run_dir):
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)


class Prompt(orm.Table):
    prompt_id: str = orm.make_field(orm.ColumnRequired(sa.String(len("005695-0057"))), default=omegaconf.MISSING)
    text: str = orm.make_field(orm.ColumnRequired(sa.String(len("005695-0057"))), default=omegaconf.MISSING)


class Conf(orm.InheritableTable):
    root_dir: str = field(default=str(DIR_ROOT.resolve()))
    out_dir: str = field(default=str((DIR_ROOT/'outputs').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='output')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)

    model_name: str = orm.make_field(orm.ColumnRequired(sa.String(len("CompVis/stable-diffusion-v1-4"))) ,default="CompVis/stable-diffusion-v1-4")
    num_inference_steps: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=100)

    resample_frequency: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=20)
    resample_t_start: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=20)
    resample_t_end: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=80)
    num_particles: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4)

    use_smc: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    prompt = orm.OneToManyField(Prompt, default=omegaconf.MISSING, enforce_element_type=False)

    lambduh: float = orm.make_field(orm.ColumnRequired(sa.Double), default=10.)
    eta: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)

    guidance_reward_fn: str = orm.make_field(orm.ColumnRequired(sa.String(len("ImageReward"))), default="ImageReward")
    potential: str = orm.make_field(orm.ColumnRequired(sa.String(len("max"))), default="max")

    def __post_init__(self):
        if self.use_smc:
            assert self.resample_frequency > 0
            assert self.num_particles > 0

    @staticmethod
    def transform_prompt(session, prompt_id):
        if prompt_id == omegaconf.MISSING:
            raise ValueError('Please specify a prompt id with prompt=<prompt_id>.')
        prompt = session.query(Prompt).filter_by(prompt_id=prompt_id).first()
        assert prompt is not None
        return prompt

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


sa.event.listens_for(Conf, 'before_insert', propagate=True)(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


orm.store_config(Conf)
orm.store_config(Prompt)
