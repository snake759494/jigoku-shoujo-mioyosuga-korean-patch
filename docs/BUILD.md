# 재빌드

## 준비
- Python 3.13, `numpy pillow opencv-python scikit-image pycdlib av imageio-ffmpeg`
- 저장소 최상위에 원본 `Jigoku Shoujo Mioyosuga (Japan).iso`(README의 해시와 일치)
- 저장소 최상위에 글꼴 `SeoulHangangL/M/B/EB.ttf`, `NanumSquareNeo-aLt~eHv.ttf`
- xdelta3 3.1.0

## 순서
```
python tools/extract_text.py          # (선택) translation/jp 재생성
python tools/check_text.py            # 번역 폭·줄 수·문자 검사 (오류 0건이어야 함)
python tools/unify.py                 # 같은 원문 = 같은 번역 통일
# 이미지: extract/img_orig/ 원본 PNG가 필요 (tools/imglib.py 로 PTD000에서 추출)
python tools/img_specs/menus.py       # 메뉴·버튼
python tools/img_specs/options.py extract/preview/options
python tools/img_specs/charsel.py     # 캐릭터 선택 + 이름판(nameplate.py)
python tools/img_specs/story.py       # 장 카드·지옥통신 화면
python tools/img_specs/ime.py         # 이름 입력 애니메이션
# 동영상: movie/orig/MOVIE_02.PSS, MOVIE_04.PSS 를 ISO에서 꺼내 둔 뒤
python tools/movie_sub.py
python tools/build.py                 # -> Jigoku_Shoujo_Mioyosuga_KR.iso
python tools/qa_leftover.py           # 결과 ISO의 일본어 잔존 전수 검사
xdelta3 -e -9 -S none -A -s "Jigoku Shoujo Mioyosuga (Japan).iso" Jigoku_Shoujo_Mioyosuga_KR.iso Jigoku_Shoujo_Mioyosuga_PS2_KO_v1.0.xdelta
```

## 번역 자료
- `translation/jp/NNN.tsv` ↔ `translation/ko/NNN.tsv`: 스크립트 문자열 (`id<TAB>문장`, 줄바꿈 `\n`)
- `translation/elf_jp.tsv`, `translation/elf_ko.tsv`: 실행파일 문자열 (파일 오프셋 16진수)
- `translation/번역_규칙.md`, `translation/glossary.tsv`: 번역 규칙·용어
- `translation/img_survey.tsv`: 글자가 있는 이미지 조사표
- `movie/subs/*.tsv`: 동영상 자막 (`시작초<TAB>끝초<TAB>문장`)

## 검증 결과 (v1.0)
- `check_text.py`: 78개 파일 오류 0건
- `qa_leftover.py`: 스크립트 문자열 11,101개 중 일본어 잔존 0건
- xdelta 왕복: 결과 ISO SHA-1 `ac339d228a315a141ac935cedac9e42d477072c3` 일치
- 실기: PCSX2 v2.2.0에서 대사창, 이름 칸, 메뉴 확인 (사용자 스크린샷)
