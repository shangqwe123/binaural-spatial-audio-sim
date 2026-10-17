"""Template captions from structured spatial labels (zh and en).

These are deliberately plain; rewrite them with an LLM later for variety and
keep the structured labels in the manifest as the ground truth.
"""
ROOM_ZH = {'small': '在一个小房间里', 'medium': '在一个中等大小的房间里',
           'large': '在一个大厅里', 'open': '在空旷的户外'}
ROOM_EN = {'small': 'In a small room', 'medium': 'In a medium-sized room',
           'large': 'In a large hall', 'open': 'Outdoors in an open space'}
SECTOR_ZH = {'front': '正前方', 'front_left': '左前方', 'left': '左侧', 'back_left': '左后方',
             'back': '正后方', 'back_right': '右后方', 'right': '右侧', 'front_right': '右前方'}
SECTOR_EN = {'front': 'the front', 'front_left': 'the front left', 'left': 'the left',
             'back_left': 'the back left', 'back': 'the back', 'back_right': 'the back right',
             'right': 'the right', 'front_right': 'the front right'}
DIST_ZH = {'near': '近处', 'far': '远处'}
DIST_EN = {'near': 'nearby', 'far': 'far away'}


def content(source):
    text = source['text'].strip()
    if source['type'] == 'speech':
        return f'一个人在说话，内容是“{text}”', f'a person says "{text}"'
    en = text.rstrip('.')
    return f'声音：{en}', en[:1].lower()+en[1:]


def make_captions(source, room_kind, spatial):
    """spatial: dict with motion, start/end {sector, distance_class}, via_sector, radial."""
    c_zh, c_en = content(source)
    r_zh, r_en = ROOM_ZH[room_kind], ROOM_EN[room_kind]
    s0, s1 = spatial['start'], spatial['end']
    m = spatial['motion']
    if m == 'static':
        zh = f'{r_zh}，{SECTOR_ZH[s0["sector"]]}{DIST_ZH[s0["distance_class"]]}，{c_zh}。'
        en = f'{r_en}, {DIST_EN[s0["distance_class"]]} from {SECTOR_EN[s0["sector"]]}, {c_en}.'
    elif m == 'radial':
        side_zh, side_en = SECTOR_ZH[s0['sector']], SECTOR_EN[s0['sector']]
        if spatial['radial'] == 'approach':
            zh = f'{r_zh}，{c_zh}，声音从{side_zh}远处逐渐靠近。'
            en = f'{r_en}, {c_en}, approaching from far away at {side_en}.'
        else:
            zh = f'{r_zh}，{c_zh}，声音从{side_zh}近处逐渐远去。'
            en = f'{r_en}, {c_en}, moving away from nearby at {side_en}.'
    else:
        d_zh, d_en = DIST_ZH[s0['distance_class']], DIST_EN[s0['distance_class']]
        a, b = SECTOR_ZH[s0['sector']], SECTOR_ZH[s1['sector']]
        ae, be = SECTOR_EN[s0['sector']], SECTOR_EN[s1['sector']]
        if m == 'arc':
            via, via_e = SECTOR_ZH[spatial['via_sector']], SECTOR_EN[spatial['via_sector']]
            zh = f'{r_zh}，{c_zh}，声音在{d_zh}从{a}经{via}绕到{b}。'
            en = f'{r_en}, {c_en}, circling {d_en} from {ae} via {via_e} to {be}.'
        else:
            zh = f'{r_zh}，{c_zh}，声音在{d_zh}从{a}移动到{b}。'
            en = f'{r_en}, {c_en}, moving {d_en} from {ae} to {be}.'
    return {'zh': zh, 'en': en}
