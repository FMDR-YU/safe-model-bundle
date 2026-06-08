# koges_model_export_study

폐쇄형 의료데이터 플랫폼의 모델 반출 거버넌스 논문용 PoC. **실데이터 미사용** — 파이프라인이 KoGES 변수 스키마만 모사해 합성데이터를 자체 생성하고, 그 위에서 PoC(D1~D4)를 실행해 Fig3·TableD0~D3을 산출한다.

## 먼저 읽기
- `claude.md` — 프로젝트 컨텍스트·규약·스코프.
- `docs/HANDOFF_koges_validation.md` — 실행 지침 + Claude Code 프롬프트(§8).
- `docs/manuscript.md` — 논문 통합 초안.

## 폴더
- `poc/` — 파이프라인(config·run_all·src). 설계·게이트 정의 불변. 산출물은 `poc/outputs/`.
- `results/` — 공유용 Fig/Table·results 노트.

## 실행
1. `cd poc && pip install -r requirements.txt`  (또는 루트에서 `bash setup.sh` / `setup.ps1`)
2. `python run_all.py`  → D1(합성)~D4(3-gate), 산출물 생성.
3. Claude Code로 자동 진행 시: `docs/HANDOFF_koges_validation.md` §8 프롬프트 붙여넣기.

## 가드레일
실제 KoGES 레코드는 쓰지 않는다(자체 합성). realism 보강이 필요하면 공개 집계 통계만 사용. 모델 바이너리는 커밋 제외.
