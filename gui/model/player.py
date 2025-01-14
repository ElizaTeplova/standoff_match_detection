class Player:
    def __init__(self, nickname, active_score=0, helper_score=0, passive_score=0):
        self.nickname = nickname
        self.active_score = active_score
        self.helper_score = helper_score
        self.passive_score = passive_score

    def inc_active(self):
        self.active_score += 1

    def inc_helper(self):
        self.helper_score += 1

    def inc_passive(self):
        self.passive_score += 1

    def __str__(self):
        return (f'nickname: {self.nickname} '
                f'active_score: {self.active_score} '
                f'helper_score: {self.helper_score} '
                f'passive_score: {self.passive_score}')
