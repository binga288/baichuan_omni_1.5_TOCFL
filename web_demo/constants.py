MODEL_PATH = "/root/.cache/huggingface/hub/models--baichuan-inc--Baichuan-Omni-1d5/snapshots/0b86202f48ec5e273e1aef3b67caf0f4e7cca1b0"
COSY_VOCODER = "./cosy24k_vocoder"
g_cache_dir = "../cache"
sampling_rate = 24000
wave_concat_overlap = int(sampling_rate * 0.01)
role_prefix = {
    'system': '<B_SYS>',
    'user': '<C_Q>',
    'assistant': '<C_A>',
    'audiogen': '<audiotext_start_baichuan>'
}
max_frames = 8
