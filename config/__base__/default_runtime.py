checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
