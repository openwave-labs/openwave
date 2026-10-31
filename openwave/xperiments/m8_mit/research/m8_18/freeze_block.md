# M8.18: the freeze block

## The terms

- The filed terms, `run/m8_16_task_details.md`: SHA-256 `924e7ee363a756a88ba2a9ee9bc03eb56c4fe27778c6e8f920a42a5229034a4c`.
- The go: https://github.com/openwave-labs/openwave/pull/687#issuecomment-6548866246
- `run/FROZEN.json` written 2026-10-31T21:13:28Z.

## The instrument

- The solver commit: MIT `4bd8bbd8e3c60a387210c4464bf0c7915d10bdc3`.
- The manifest: 37 files, SHA-256 `6d244f4af56b71d60f17077724628b7489b278e84a7d24565c17038ed19ba43a`; `run/MANIFEST_PIN` holds `6d244f4af56b71d60f17077724628b7489b278e84a7d24565c17038ed19ba43a`.
- The freeze pin, `run/FROZEN_PIN`: `b5902494f927ec5daee1562fdf3d64bf3f402eb2cc9cac777de7a29aaa7c32d5`.

| file | SHA-256 |
| --- | --- |
| `instrument/__init__.py` | `ebb8d0440edae25ee0755fa962d24061fbebb502bf5176d3084a422188d2dcdc` |
| `instrument/basis.py` | `5728c5e8c9532bc4764dd5b9d111f693e5ba86c415d4a2b4f880e8196c66616b` |
| `instrument/branches.py` | `4e32a8cfd1cd7db4ced242ecddf5e22c9319134029877b78eae74b13b3d1a55f` |
| `instrument/certified.py` | `1f5dde36bde7edfc3257e7581b2538ab7853693ab8ed1b48dae6dc0e2de343d1` |
| `instrument/convergence.py` | `02404ab43313eb4d67cb336edbed364d9071187da66dec44bbd4f997cf66bd51` |
| `instrument/devtools.py` | `3a2d0f7d2c5bd74bdedf2a673989a67cadef25c508424ad25501745c7adb7d0f` |
| `instrument/functional.py` | `f8f9329f1fa5d2bb1f7e3718d5e5bdfa94e86f9e0dcc4dda308a7f86c0f00d79` |
| `instrument/group2i.py` | `0a9732afe32b3509976de2e519c0f8da3d4f773d2495c5c2d58d27fe5aaa85f0` |
| `instrument/linear.py` | `e0fce88f3ed8374f94f6ba329813cfb639f05a2407f9a313f66cbe79744f11c2` |
| `instrument/newton.py` | `36bd989fe10891ba49757025a908b95569cb7b7ee1901f87299dede2a10fe00f` |
| `instrument/quadrature.py` | `6c9baf9d1bd3bc8741006490f88fd4a1e134f0cd3a6ffd8f9b7f1a6074e30a71` |
| `instrument/su2.py` | `60c4f8088a089341e76fb159847784336f4ad6a886a72a1c7683ef86a63b75f6` |
| `run/ENVIRONMENT.json` | `79f009089a3b983d9f0e48e3e80dc0d17bc354520d20c332fc2cf7274eff6e80` |
| `run/cases.py` | `cfc3663dc5c9ac469335ca71d8884312504362c455b6291b758d45b92ab25e9c` |
| `run/census.py` | `6000ecc7dbd570b0bb6fee6493ff01634bcca1e94a350e4a46c2483d109cb8e3` |
| `run/census_lib.py` | `509b562200d9faeae02b730b4cd97a646ba3ccc4d6790d41ac04f0357bb86857` |
| `run/checks.py` | `c3122cd0ac692e16a5a6fb7fde178e13b27d9bb1632ad8eed32432b2c6919b84` |
| `run/driver.py` | `6fad387effeb2ac805a09f485a11037b7575f511558776d0461dd8c6f5b622f9` |
| `run/freeze_block.py` | `70d2cfd85c32490ab25c2ac99122cbc9e847befa443bb91e892aaeb590cea89f` |
| `run/m8_16_task_details.md` | `924e7ee363a756a88ba2a9ee9bc03eb56c4fe27778c6e8f920a42a5229034a4c` |
| `run/next_order_values.json` | `b05fff5ddd71c29c79318b6257f940c6035d7c954f3bafa34a22080eecc00333` |
| `run/pin_manifest.py` | `4e7c7ab3eb36bafcb956c67751bdbc121767ba9b57bec89a8a108f5b67798580` |
| `run/record_environment.py` | `1494fb8afd942e74a646671fe16f03ad4d9b095ae62cd1be35a51b2eb3a10059` |
| `run/seeds.py` | `11916c7ccd9c27583933c78b8d79f5c9ed0472c4ade946312acf4cf0bd80d87c` |
| `run/test_constants.out` | `a46363513bf127b4325d18e343e5661e4e67b3a517aa14cc255b154068c7b34b` |
| `run/test_constants.py` | `7bffcf90f8021ef54f8783f53a0a55acea9973526a559d32ab8069825dc29fc8` |
| `run/test_freeze.out` | `a93e3b86031bae5660dd045979af698102a43cf434255d4b010aedfbbf259de6` |
| `run/test_freeze.py` | `7d3207c51a51089ca52be3eefdcac03a874e7827c8a3b959a860fbb18219cb4f` |
| `run/test_gate.out` | `745378fdae6481815d49e88d53ce5bc662a5bf74dabb8954018316f70eab2d13` |
| `run/test_gate.py` | `3ef342b7a19fcd2a8df3eb258126662db675b2426d825d5bbdd634637488efe5` |
| `run/test_locator.out` | `3ed2e6d7832bfe885cdaf3e265a437ea6d2bb1b5992e672fc2089e19f34a6dab` |
| `run/test_locator.py` | `9e62381164f1d2ddfe76892fe05ed41261d991d341f642314777cb880e09e351` |
| `run/test_reader.out` | `bebbf128c339a3fcce1d0fd0bafce993da2cec3bd5b12650b66db0145e5757e4` |
| `run/test_reader.py` | `83d909fc44018b96a22b6f02003c1b3e08b5fc1e486daf8c40f9228f00d664fa` |
| `run/test_scope.out` | `d9e084076c93413ad8fa398e8cfe2c6b234c712768a567316aaf76bc184e6426` |
| `run/test_scope.py` | `43a928cd87d540ca86df0fdd4bde37a70094a4e385eccfdd75d863eb3a6aa846` |
| `run/write_guard.py` | `b971bd13fc351f2ed1485878fcdac7da11fbc29c21cf137b8450119ebfb1d7a2` |

## The environment

- Recorded 2026-10-31T10:07:21Z, and read again when this block was built.
- Differences: none.

<details><summary>As recorded</summary>

```json
{
 "interpreter": {
  "build": "3.13.13 | packaged by conda-forge | (main, Apr  8 2026, 02:29:07) [Clang 19.1.7 ]",
  "executable": "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
  "flags": {
   "dont_write_bytecode": 1,
   "isolated": 1,
   "optimize": 0
  },
  "implementation": "CPython",
  "version": "3.13.13"
 },
 "machine": {
  "arch": "arm64",
  "cores": 8,
  "cpu": "Apple M3",
  "efficiency_cores": 4,
  "model": "Mac15,13",
  "performance_cores": 4
 },
 "numpy": {
  "blas": {
   "detection method": "system",
   "found": true,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "lapack": {
   "detection method": "system",
   "found": true,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "version": "2.5.0"
 },
 "os": {
  "build": "24H232",
  "kernel": "24.6.0",
  "product": "macOS",
  "version": "15.8.2"
 },
 "recorded": "2026-10-31T10:07:21Z",
 "scipy": {
  "blas": {
   "cython blas ilp64": false,
   "detection method": "extraframeworks",
   "found": true,
   "has ilp64": false,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "Accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "lapack": {
   "detection method": "extraframeworks",
   "found": true,
   "has ilp64": false,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "Accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "version": "1.18.0"
 },
 "threading": {
  "MKL_NUM_THREADS": null,
  "OMP_NUM_THREADS": null,
  "OPENBLAS_NUM_THREADS": null,
  "VECLIB_MAXIMUM_THREADS": null
 }
}
```

</details>

<details><summary>Read again</summary>

```json
{
 "interpreter": {
  "build": "3.13.13 | packaged by conda-forge | (main, Apr  8 2026, 02:29:07) [Clang 19.1.7 ]",
  "executable": "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
  "flags": {
   "dont_write_bytecode": 1,
   "isolated": 1,
   "optimize": 0
  },
  "implementation": "CPython",
  "version": "3.13.13"
 },
 "machine": {
  "arch": "arm64",
  "cores": 8,
  "cpu": "Apple M3",
  "efficiency_cores": 4,
  "model": "Mac15,13",
  "performance_cores": 4
 },
 "numpy": {
  "blas": {
   "detection method": "system",
   "found": true,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "lapack": {
   "detection method": "system",
   "found": true,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "version": "2.5.0"
 },
 "os": {
  "build": "24H232",
  "kernel": "24.6.0",
  "product": "macOS",
  "version": "15.8.2"
 },
 "scipy": {
  "blas": {
   "cython blas ilp64": false,
   "detection method": "extraframeworks",
   "found": true,
   "has ilp64": false,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "Accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "lapack": {
   "detection method": "extraframeworks",
   "found": true,
   "has ilp64": false,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "Accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "version": "1.18.0"
 },
 "threading": {
  "MKL_NUM_THREADS": null,
  "OMP_NUM_THREADS": null,
  "OPENBLAS_NUM_THREADS": null,
  "VECLIB_MAXIMUM_THREADS": null
 }
}
```

</details>

## The reference files

| name | MIT commit | blob | SHA-256 |
| --- | --- | --- | --- |
| reference/m8_16_1/run/out/RUNLOG.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `ccdd58fe8a4498b1d94ebf272ce2c80aaafb193e` | `84f2c9f6667e02d030dfe1d0f759ad183cd0bc6293228bd60e3aa0a2785edadb` |
| reference/m8_16_1/run/out/C1.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `4ccac3d2843c4082fb13af7d34ece7b49c15fe07` | `356ffbf4fe71eb1429163e849d53a33f92643b5784124a9d11e9ef8c5e9b15ca` |
| reference/m8_16_1/run/out/C2a.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `91564933c228cdc5df300decaae4e62602d42b4b` | `0165455b6ba66d462888521a5273fc8f0a744d4430c808e2855f5e9d06d3c24f` |
| reference/m8_16_1/run/out/C2b.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `19f80bada697e9d7e3a9274270089cb2e0c8cc1c` | `efd5c21e233f08897f5dc9ded449801bfa3fe8d5e8895b6099b1840dd2b5a0fd` |
| reference/m8_16_1/run/out/C3.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `8ebe57cbb36e2e9704f7bd6208901311e5b82a70` | `8e7c303c2e7dfe35e39e3f581ace30d537a1f30073389f8e6aa199b30250d469` |
| reference/m8_16_1/run/out/C2a_R5.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `af6c5fce0d2befbbade54a375c7db89e391226a2` | `387ed7b8e04030a2e10569eedb313ca5713bea92bec5a6bac7bd2e6697e668be` |
| reference/m8_16_1/run/out/P5.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `b123611e0b013fadf9d449ee90302f6924ebbf6a` | `c404e49c2a6e52b0754c97349b01c8097db3271af641bc56071fd72098082bd0` |
| reference/m8_16_1/run/out/C3R_R4.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `dfc8566b1f1969030be5b5161f17b83fdd16e2f5` | `21706fd5b2548cdcb0bf8b44d2df0c6b62d34abad41363d0fb1e43a14a5c9fb6` |
| reference/m8_16_1/run/FROZEN.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `9da1269b789bc8703d0ec819657b70bcffae1f9c` | `c258e299b8f0268e805712b4f087b3533c6e0cd3c37e237bf3d0639540c52a2d` |
| reference/m8_16_1/run/m8_16_task_details.md | `292b158cf1a53902489ced3426e1f047699aeb62` | `746fb0c38ceb994caf9f32096d455e63c1f4a80b` | `9d16712982e5388970fca33de59958721109e09f9d88dec7a2a088489ad3efd6` |
| reference/m8_17/run/out/RUNLOG.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `d3e29ce7d6f5ce64b721e17b188c72db2e174b64` | `7d761506f8bfb05cee50ddc6cbf29b0a97fb20c7d9242d57c6765cff8114196a` |
| reference/m8_17/run/out/T1_R4.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `d7480d323c91df14a990aea1cfe3a7f4751d53f9` | `06e51a090e310a2accf6bab21fdee693ab2806ff6115bb8a7f10b54aae96bcc9` |
| reference/m8_17/run/out/T1_R5.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `1740687e6215cc0783defdb279db939ef3a3cf95` | `e8ecd65f3754e6bd047e0d6c34964d59aaa44bd05bb522a71a884118926d3678` |
| reference/m8_17/run/out/T4_R4.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `2e825c81b3e8dcb52daf25fbfd0f4b09a80f713c` | `2955bc02cb2c653deb28f13bcdf53229b21541521681900661183a632823e512` |
| reference/m8_17/run/out/T4_R5.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `1cb6cee89f8dc653f038e6c894336ceb9349d656` | `167ddab0d696d2758581a8f7063d0d74283c13bbe2500fb60c91665e1a2ce95c` |
| reference/m8_17/run/out/T5_R2.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `5501e2c7590788242ba5da7bb8d07cf2710bc72c` | `fb185a91edcf7099950e4e7529d125136537769992d8bddb9cb18edf0d632bec` |
| reference/m8_17/run/out/locate_T4_R5_8_11.313708499.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `cee35076c3670febf1f4e08f9baeea571d89ff20` | `517aa9f62a45255ec4a76690e95069935d32265e532a7858e58d8709cf10af55` |
| reference/m8_17/run/out/CENSUS.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `626c25386b93a36f9b7c5bd9dd5215e1822c05f9` | `d010db400a6bdef2ad64ce1bfd0e085995c49e62ba1982325ff6234f8d233aac` |
| reference/m8_17/run/census.py | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `7be181f99c814cd422ffdb34d80947fa9a9aa5c0` | `f4c60943215bea341656638ec431ee79206d0561fd266906b018196cfdb5421d` |
| reference/m8_17/run/census_lib.py | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `2889ff860096c8e610115bdc39a085e5e8619fac` | `4873006e94e09bdf1b03396e2e9e85881ce503d2def0a2036858691706d6a321` |
| reference/m8_17/run/m8_16_task_details.md | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `c6b2b13a88b59734ac949d18541f5490ea49332d` | `667159c3db993396a10a75775acf79e78f744d161c641ebc37c9e6baf3af7cea` |
| SOURCES.json | `this tree` | `` | `c90e734e8817423ed8053475072dc44efa2fc76db7cec974e1321be421a4da31` |

## Inherited from M8.16.1

- Its filed terms: SHA-256 `9d16712982e5388970fca33de59958721109e09f9d88dec7a2a088489ad3efd6`.
- Its manifest pin: `8cc2d7e2d01a72fc3572adabf917656abf36cbddd70ed35532b73ae5451790dd`.
- Its go: https://github.com/openwave-labs/openwave/pull/650#issuecomment-6385959498
- Its pinned next-order values, `run/next_order_values.json`: SHA-256 `b05fff5ddd71c29c79318b6257f940c6035d7c954f3bafa34a22080eecc00333`; its cases section `45ad237179b6c7176b4372f8b36fe6cf20ba95ebdff510f727eec1ea4a2ebeea`.

## The census values

- Arm A's ladder: 24, 32, 40, a location decided at 32, 40.
- Arm B's ladder: 16, 24, 32, its location decided at 24, 32.
- The band: n_lo = 16, levels {'R2': 17, 'R4': 18, 'R5': 18}; the unscored reading beside it, n_lo = 24.
- The moved midpoint: 0.4 of the interval above its lower end, on T4/R5's (8, 11.3137) alone.
- The locator: w = 0.001, each cutoff's bisection to 0.0005; the guards at 0.01, 0.05; the two-ends agreement 0.01.
- The twelve: 12 points.
- The controls: C1, C2a, C2b, C3, C2a/R5, P5, C3R/R4.

## `FROZEN.json`

<details><summary>In full</summary>

```json
{
 "allowlist": [
  "T4/R5",
  "C1",
  "C2a",
  "C2b",
  "C3",
  "C2a/R5",
  "P5",
  "C3R/R4",
  "T5/R2",
  "T1/R4",
  "T4/R4",
  "T1/R5"
 ],
 "order": [
  "T4/R5",
  "C1",
  "C2a",
  "C2b",
  "C3",
  "C2a/R5",
  "P5",
  "C3R/R4",
  "T5/R2",
  "T1/R4",
  "T4/R4",
  "T1/R5"
 ],
 "cases": {
  "T4/R5": {
   "role": "row",
   "slot": "R5",
   "level": 6,
   "lam0": 48.0,
   "orbit": "outside3",
   "sig": [
    7,
    0,
    2
   ],
   "k": 4,
   "r": 2,
   "points": [
    {
     "eps": 0.25,
     "read": true
    },
    {
     "eps": 0.3535533905932738,
     "read": true
    },
    {
     "eps": 0.5,
     "read": true
    },
    {
     "eps": 0.7071067811865476,
     "read": true
    },
    {
     "eps": 1.0,
     "read": true
    },
    {
     "eps": 1.4142135623730951,
     "read": true
    },
    {
     "eps": 2.0,
     "read": true
    },
    {
     "eps": 2.8284271247461903,
     "read": true
    },
    {
     "eps": 4.0,
     "read": true
    },
    {
     "eps": 5.656854249492381,
     "read": true
    },
    {
     "eps": 8.0,
     "read": true
    },
    {
     "eps": 11.313708498984761,
     "read": true
    }
   ],
   "cutoffs": [
    16,
    24,
    32
   ],
   "m815": null,
   "dim": 6,
   "ranks": [
    2,
    0
   ],
   "chains": 4,
   "type": "Elliptic",
   "idx": [
    7,
    1,
    6
   ],
   "kr": 0,
   "kc": 0,
   "ki": 3,
   "next_order": {
    "16": [
     {
      "s": [
       "-7.160185136497685502059357e-52",
       "0.003226983312990572834385275"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-1.169970658695319451313197e-53",
       "0.00005272874542103126692183694"
      ],
      "c_g": [
       "1.329344481921162342070699e-53",
       "0.00001232616058683281576614424"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     },
     {
      "s": [
       "6.139054330132819221255153e-52",
       "0.07832915604244644610284651"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.003118397285615959095325e-53",
       "0.00127989447958402176448166"
      ],
      "c_g": [
       "1.980680386313032911772464e-54",
       "-0.00007549196435959471550789545"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "2.308743382967845127757687e-51",
       "0.1747646675783438191780421"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "3.772475103696256329082133e-53",
       "0.002855645899448308927242345"
      ],
      "c_g": [
       "2.652896615578911939699727e-54",
       "0.0000487557438198699684901474"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-9.8933018395933414729422e-52",
       "0.5693015277529908523470826"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-1.616560556645374794964708e-53",
       "0.009302358398894923571101506"
      ],
      "c_g": [
       "1.270910195992155396089878e-54",
       "-0.0001835569882328996141696661"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     }
    ],
    "24": [
     {
      "s": [
       "-7.160185136497685502059357e-52",
       "0.003226983312990572834385275"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-1.169970658695319451313197e-53",
       "0.00005272874542103126692183694"
      ],
      "c_g": [
       "1.408906032540104615468062e-53",
       "0.00001265437152491790586718242"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     },
     {
      "s": [
       "6.139054330132819221255153e-52",
       "0.07832915604244644610284651"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.003118397285615959095325e-53",
       "0.00127989447958402176448166"
      ],
      "c_g": [
       "2.125690572063365584713881e-54",
       "-0.0000788738631454683320333208"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "2.308743382967845127757687e-51",
       "0.1747646675783438191780421"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "3.772475103696256329082133e-53",
       "0.002855645899448308927242345"
      ],
      "c_g": [
       "2.429355284444535885706282e-54",
       "0.00004098350169631699511132212"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-9.8933018395933414729422e-52",
       "0.5693015277529908523470826"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-1.616560556645374794964708e-53",
       "0.009302358398894923571101506"
      ],
      "c_g": [
       "1.209516778420864361610713e-54",
       "-0.0002149395213634538703315838"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     }
    ],
    "32": [
     {
      "s": [
       "-7.160185136497685502059357e-52",
       "0.003226983312990572834385275"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-1.169970658695319451313197e-53",
       "0.00005272874542103126692183694"
      ],
      "c_g": [
       "1.408906032540104615468062e-53",
       "0.00001265437152491790586718242"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     },
     {
      "s": [
       "6.139054330132819221255153e-52",
       "0.07832915604244644610284651"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.003118397285615959095325e-53",
       "0.00127989447958402176448166"
      ],
      "c_g": [
       "2.125690572063365584713881e-54",
       "-0.0000788738631454683320333208"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "2.308743382967845127757687e-51",
       "0.1747646675783438191780421"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "3.772475103696256329082133e-53",
       "0.002855645899448308927242345"
      ],
      "c_g": [
       "2.429355284444535885706282e-54",
       "0.00004098350169631699511132212"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-9.8933018395933414729422e-52",
       "0.5693015277529908523470826"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-1.616560556645374794964708e-53",
       "0.009302358398894923571101506"
      ],
      "c_g": [
       "1.209516778420864361610713e-54",
       "-0.0002149395213634538703315838"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     }
    ]
   },
   "Q": {
    "value": 1.104176947003101
   },
   "s": {
    "frac": "9/16",
    "value": 0.5625
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "value": 1.01209160489
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 0.310692742362
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 0.139251832964
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 0.00573685922309
     },
     "mult": 1
    }
   ],
   "representative": {
    "3": [
     -0.3086472568785893,
     -0.06949329411780472
    ],
    "2": [
     0.34579154134174556,
     -0.37948776525885436
    ],
    "1": [
     0.20302636427665244,
     0.1632131521070081
    ],
    "0": [
     0.29228686825253675,
     0.09882806648613832
    ],
    "-1": [
     0.3588169781640245,
     -0.4475251820209991
    ],
    "-2": [
     0.0005102297941505528,
     0.0441294932645358
    ],
    "-3": [
     0.22297893474867425,
     -0.30425691601898186
    ]
   },
   "admission": "T4/R5"
  },
  "C1": {
   "role": "control",
   "slot": "R3",
   "level": 2,
   "lam0": 8.0,
   "orbit": "theoremD",
   "sig": null,
   "k": 4,
   "r": 2,
   "points": [
    {
     "eps": 0.5,
     "read": true
    },
    {
     "eps": 1.0,
     "read": true
    },
    {
     "eps": 2.0,
     "read": true
    }
   ],
   "cutoffs": [
    16,
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 6,
   "ranks": [
    1,
    0
   ],
   "chains": 5,
   "type": "Elliptic",
   "idx": [
    0,
    0,
    0
   ],
   "kr": 0,
   "kc": 0,
   "ki": 0,
   "Q": {
    "frac": "1/1",
    "value": 1.0
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [],
   "representative": {
    "1": [
     1.0,
     0.0
    ],
    "0": [
     0.6,
     0.3
    ],
    "-1": [
     0.0,
     -0.4
    ]
   },
   "admission": "theoremD:R3"
  },
  "C2a": {
   "role": "control",
   "slot": "R4",
   "level": 6,
   "lam0": 48.0,
   "orbit": "coherent",
   "sig": [
    0,
    0,
    10
   ],
   "k": 3,
   "r": 1,
   "points": [
    {
     "eps": 0.25,
     "read": true
    },
    {
     "eps": 0.5,
     "read": true
    },
    {
     "eps": 1.0,
     "read": true
    },
    {
     "eps": 2.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 4,
   "ranks": [
    1,
    0
   ],
   "chains": 3,
   "type": "Elliptic",
   "idx": [
    0,
    0,
    0
   ],
   "kr": 0,
   "kc": 0,
   "ki": 0,
   "Q": {
    "frac": "1288/1287",
    "value": 1.0007770007770007
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "28/143",
      "value": 0.1958041958041958
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "56/429",
      "value": 0.13053613053613053
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "140/99",
      "value": 1.4141414141414141
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "112/39",
      "value": 2.871794871794872
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "896/1287",
      "value": 0.6961926961926962
     },
     "mult": 1
    }
   ],
   "representative": {
    "3": [
     1.0,
     0.0
    ]
   },
   "admission": "C2a"
  },
  "C2b": {
   "role": "control",
   "slot": "R2",
   "level": 7,
   "lam0": 63.0,
   "orbit": "weight72",
   "sig": [
    0,
    0,
    10
   ],
   "k": 5,
   "r": 1,
   "points": [
    {
     "eps": 0.25,
     "read": true
    },
    {
     "eps": 0.5,
     "read": true
    },
    {
     "eps": 1.0,
     "read": true
    },
    {
     "eps": 2.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 6,
   "ranks": [
    1,
    0
   ],
   "chains": 5,
   "type": "Elliptic",
   "idx": [
    0,
    0,
    0
   ],
   "kr": 0,
   "kc": 0,
   "ki": 0,
   "Q": {
    "frac": "144/143",
    "value": 1.006993006993007
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "112/143",
      "value": 0.7832167832167832
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "168/143",
      "value": 1.1748251748251748
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "336/143",
      "value": 2.3496503496503496
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "504/143",
      "value": 3.5244755244755246
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "560/143",
      "value": 3.9160839160839163
     },
     "mult": 1
    }
   ],
   "representative": {
    "3.5": [
     1.0,
     0.0
    ]
   },
   "admission": "C2b"
  },
  "C3": {
   "role": "control",
   "slot": "R4",
   "level": 6,
   "lam0": 48.0,
   "orbit": "octahedron",
   "sig": [
    6,
    0,
    3
   ],
   "k": 4,
   "r": 4,
   "points": [
    {
     "eps": 0.25,
     "read": true
    },
    {
     "eps": 0.5,
     "read": true
    },
    {
     "eps": 1.0,
     "read": true
    },
    {
     "eps": 2.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 8,
   "ranks": [
    4,
    0
   ],
   "chains": 4,
   "type": "Hyperbolic",
   "idx": [
    6,
    3,
    3
   ],
   "kr": 3,
   "kc": 0,
   "ki": 0,
   "Q": {
    "frac": "175/143",
    "value": 1.2237762237762237
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [
    {
     "cls": "real",
     "sign": null,
     "tau": {
      "value": 0.674085679526388
     },
     "mult": 3
    }
   ],
   "representative": {
    "2": [
     1.0,
     0.0
    ],
    "-2": [
     1.0,
     0.0
    ]
   },
   "admission": "C3"
  },
  "C2a/R5": {
   "role": "R5gate",
   "slot": "R5",
   "level": 6,
   "lam0": 48.0,
   "orbit": "coherent",
   "sig": [
    0,
    0,
    10
   ],
   "k": 3,
   "r": 1,
   "points": [
    {
     "eps": 0.25,
     "read": true
    },
    {
     "eps": 0.5,
     "read": true
    },
    {
     "eps": 1.0,
     "read": true
    },
    {
     "eps": 2.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 4,
   "ranks": [
    1,
    0
   ],
   "chains": 3,
   "type": "Elliptic",
   "idx": [
    0,
    0,
    0
   ],
   "kr": 0,
   "kc": 0,
   "ki": 0,
   "Q": {
    "frac": "2289/2288",
    "value": 1.000437062937063
   },
   "s": {
    "frac": "9/16",
    "value": 0.5625
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "28/143",
      "value": 0.1958041958041958
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "56/429",
      "value": 0.13053613053613053
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "140/99",
      "value": 1.4141414141414141
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "112/39",
      "value": 2.871794871794872
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "896/1287",
      "value": 0.6961926961926962
     },
     "mult": 1
    }
   ],
   "representative": {
    "3": [
     1.0,
     0.0
    ]
   },
   "admission": "C2a/R5"
  },
  "P5": {
   "role": "pyramid",
   "slot": "R4",
   "level": 6,
   "lam0": 48.0,
   "orbit": "pyramid",
   "sig": [
    5,
    0,
    4
   ],
   "k": 4,
   "r": 2,
   "points": [
    {
     "eps": 0.25,
     "read": true
    },
    {
     "eps": 0.5,
     "read": true
    },
    {
     "eps": 1.0,
     "read": true
    },
    {
     "eps": 2.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 6,
   "ranks": [
    2,
    0
   ],
   "chains": 4,
   "type": "Hyperbolic",
   "idx": [
    5,
    1,
    4
   ],
   "kr": 0,
   "kc": 1,
   "ki": 1,
   "next_order": {
    "16": [
     {
      "s": [
       "-1.643269893669447657559704e-51",
       "0.06223486909568790724544574"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-2.502769906858040393003423e-53",
       "0.0009478635136564426746242348"
      ],
      "c_g": [
       "4.916489626638153205146562e-55",
       "-0.00001064621147000250005841326"
      ],
      "c_g_off": null,
      "under_bar": false,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.04173528598376259430139185",
       "0.06091686091686091686091686"
      ],
      "mult": 1,
      "cls": "complex",
      "krein": null,
      "kappa_s": [
       "0.0006356461480653562023710027",
       "0.0009277896887803903470073557"
      ],
      "c_g": [
       "0.0001195515456890397380967572",
       "-0.0002358143032732839336240766"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-4.902349205570723022598736e-80",
       "1.095185488324669513111975"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-7.466486249081060140228799e-82",
       "0.01668014057317097391851552"
      ],
      "c_g": [
       "6.764232268557010721252782e-56",
       "-0.0000767611760516291178604035"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     }
    ],
    "24": [
     {
      "s": [
       "-1.643269893669447657559704e-51",
       "0.06223486909568790724544574"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-2.502769906858040393003423e-53",
       "0.0009478635136564426746242348"
      ],
      "c_g": [
       "4.617478195520401141526899e-55",
       "-0.00001313911766073773401189807"
      ],
      "c_g_off": null,
      "under_bar": false,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.04173528598376259430139185",
       "0.06091686091686091686091686"
      ],
      "mult": 1,
      "cls": "complex",
      "krein": null,
      "kappa_s": [
       "0.0006356461480653562023710027",
       "0.0009277896887803903470073557"
      ],
      "c_g": [
       "0.0001242332455239913262308566",
       "-0.0002348453653056894878831448"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-4.902349205570723022598736e-80",
       "1.095185488324669513111975"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-7.466486249081060140228799e-82",
       "0.01668014057317097391851552"
      ],
      "c_g": [
       "9.179640923728366821050269e-56",
       "-0.0001321715189203882735231213"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     }
    ],
    "32": [
     {
      "s": [
       "-1.643269893669447657559704e-51",
       "0.06223486909568790724544574"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-2.502769906858040393003423e-53",
       "0.0009478635136564426746242348"
      ],
      "c_g": [
       "4.617478195520401141526899e-55",
       "-0.00001313911766073773401189807"
      ],
      "c_g_off": null,
      "under_bar": false,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.04173528598376259430139185",
       "0.06091686091686091686091686"
      ],
      "mult": 1,
      "cls": "complex",
      "krein": null,
      "kappa_s": [
       "0.0006356461480653562023710027",
       "0.0009277896887803903470073557"
      ],
      "c_g": [
       "0.0001242332455239913262308566",
       "-0.0002348453653056894878831448"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-4.902349205570723022598736e-80",
       "1.095185488324669513111975"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-7.466486249081060140228799e-82",
       "0.01668014057317097391851552"
      ],
      "c_g": [
       "9.179640923728366821050269e-56",
       "-0.0001321715189203882735231213"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     }
    ]
   },
   "Q": {
    "frac": "77/65",
    "value": 1.1846153846153846
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 0.0622348691
     },
     "mult": 1
    },
    {
     "cls": "complex",
     "sign": null,
     "tau": {
      "re": 0.041735286,
      "im": 0.0609168609
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "value": 1.0951854883
     },
     "mult": 1
    }
   ],
   "representative": {
    "3": [
     3.4641016151377544,
     0.0
    ],
    "-2": [
     3.605551275463989,
     0.0
    ]
   },
   "admission": "P5"
  },
  "C3R/R4": {
   "role": "ray",
   "slot": "R4",
   "level": 6,
   "lam0": 48.0,
   "orbit": "c3ray",
   "sig": [
    5,
    0,
    4
   ],
   "k": 4,
   "r": 2,
   "points": [
    {
     "eps": 0.25,
     "read": true
    },
    {
     "eps": 0.5,
     "read": true
    },
    {
     "eps": 1.0,
     "read": true
    },
    {
     "eps": 2.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 6,
   "ranks": [
    2,
    0
   ],
   "chains": 4,
   "type": "Hyperbolic",
   "idx": [
    5,
    1,
    4
   ],
   "kr": 0,
   "kc": 1,
   "ki": 1,
   "next_order": {
    "16": [
     {
      "s": [
       "0.03556672536102731006354228",
       "0.009869528771337233982631024"
      ],
      "mult": 1,
      "cls": "complex",
      "krein": null,
      "kappa_s": [
       "0.0005416963473984859819392362",
       "0.0001503171189281290789094394"
      ],
      "c_g": [
       "0.00001638557853285604182167366",
       "0.00005721327997393224340340862"
      ],
      "c_g_off": [
       "-0.0001290926453744682420226969",
       "-0.00005206109914080668025634546"
      ],
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "2.2495847514701127814879e-52",
       "0.4538302354347943530089112"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "3.426213210986183207450137e-54",
       "0.006912027418284713557779736"
      ],
      "c_g": [
       "1.635603923170307435035513e-55",
       "-0.0001091831520530414277722538"
      ],
      "c_g_off": [
       "-9.757608770173252729187422e-56",
       "-0.0003587480448487821884433321"
      ],
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-5.238487977569249393177244e-52",
       "1.001806258005316012392248"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-7.978439888788741290922846e-54",
       "0.01525793519796645992235538"
      ],
      "c_g": [
       "7.490556479532316931037989e-55",
       "-0.00002084260886184119665569045"
      ],
      "c_g_off": [
       "5.12577814233763978609055e-55",
       "-0.00009059468220624881626274189"
      ],
      "under_bar": true,
      "spectrum_index": 2
     }
    ],
    "24": [
     {
      "s": [
       "0.03556672536102731006354228",
       "0.009869528771337233982631024"
      ],
      "mult": 1,
      "cls": "complex",
      "krein": null,
      "kappa_s": [
       "0.0005416963473984859819392362",
       "0.0001503171189281290789094394"
      ],
      "c_g": [
       "0.00001576644826655185010137049",
       "0.0000569107353835280937188124"
      ],
      "c_g_off": [
       "-0.0001253558091910067680377724",
       "-0.00004909170710414556010692469"
      ],
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "2.2495847514701127814879e-52",
       "0.4538302354347943530089112"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "3.426213210986183207450137e-54",
       "0.006912027418284713557779736"
      ],
      "c_g": [
       "1.590992533560310168437763e-55",
       "-0.0001309630754965078696466127"
      ],
      "c_g_off": [
       "-9.42181750189874749671704e-56",
       "-0.000373055397684360691873811"
      ],
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-5.238487977569249393177244e-52",
       "1.001806258005316012392248"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-7.978439888788741290922846e-54",
       "0.01525793519796645992235538"
      ],
      "c_g": [
       "7.355350858404895233743971e-55",
       "-0.00006863064912597637137821637"
      ],
      "c_g_off": [
       "5.061379648361528056092418e-55",
       "-0.000136294178335535390505815"
      ],
      "under_bar": true,
      "spectrum_index": 2
     }
    ],
    "32": [
     {
      "s": [
       "0.03556672536102731006354228",
       "0.009869528771337233982631024"
      ],
      "mult": 1,
      "cls": "complex",
      "krein": null,
      "kappa_s": [
       "0.0005416963473984859819392362",
       "0.0001503171189281290789094394"
      ],
      "c_g": [
       "0.00001576644826655185010137049",
       "0.0000569107353835280937188124"
      ],
      "c_g_off": [
       "-0.0001253558091910067680377724",
       "-0.00004909170710414556010692469"
      ],
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "2.2495847514701127814879e-52",
       "0.4538302354347943530089112"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "3.426213210986183207450137e-54",
       "0.006912027418284713557779736"
      ],
      "c_g": [
       "1.590992533560310168437763e-55",
       "-0.0001309630754965078696466127"
      ],
      "c_g_off": [
       "-9.42181750189874749671704e-56",
       "-0.000373055397684360691873811"
      ],
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-5.238487977569249393177244e-52",
       "1.001806258005316012392248"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-7.978439888788741290922846e-54",
       "0.01525793519796645992235538"
      ],
      "c_g": [
       "7.355350858404895233743971e-55",
       "-0.00006863064912597637137821637"
      ],
      "c_g_off": [
       "5.061379648361528056092418e-55",
       "-0.000136294178335535390505815"
      ],
      "under_bar": true,
      "spectrum_index": 2
     }
    ]
   },
   "Q": {
    "frac": "77/65",
    "value": 1.1846153846153846
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [
    {
     "cls": "complex",
     "sign": null,
     "tau": {
      "re": 0.03556672536102731,
      "im": 0.009869528771337234
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 0.45383023543479434
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "value": 1.001806258005316
     },
     "mult": 1
    }
   ],
   "representative": {
    "2": [
     1.0,
     0.0
    ],
    "-1": [
     2.0,
     0.0
    ]
   },
   "admission": "C3R/R4"
  },
  "T5/R2": {
   "role": "row",
   "slot": "R2",
   "level": 7,
   "lam0": 63.0,
   "orbit": "line72_32",
   "sig": [
    9,
    0,
    0
   ],
   "k": 6,
   "r": 4,
   "points": [
    {
     "eps": 0.25,
     "read": false
    },
    {
     "eps": 0.3535533905932738,
     "read": false
    },
    {
     "eps": 0.5,
     "read": false
    },
    {
     "eps": 0.7071067811865476,
     "read": false
    },
    {
     "eps": 1.0,
     "read": false
    },
    {
     "eps": 1.4142135623730951,
     "read": false
    },
    {
     "eps": 2.0,
     "read": false
    },
    {
     "eps": 2.8284271247461903,
     "read": false
    },
    {
     "eps": 4.0,
     "read": false
    },
    {
     "eps": 5.656854249492381,
     "read": false
    },
    {
     "eps": 8.0,
     "read": false
    },
    {
     "eps": 11.313708498984761,
     "read": false
    },
    {
     "eps": 16.0,
     "read": false
    },
    {
     "eps": 22.627416997969522,
     "read": false
    },
    {
     "eps": 32.0,
     "read": false
    },
    {
     "eps": 45.254833995939045,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": "T5_R2",
   "dim": 10,
   "ranks": [
    4,
    0
   ],
   "chains": 6,
   "type": "Elliptic",
   "idx": [
    9,
    3,
    6
   ],
   "kr": 0,
   "kc": 0,
   "ki": 3,
   "next_order": {
    "16": [
     {
      "s": [
       "-5.297038257226489638821057e-82",
       "2.68857172135926286741336"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-4.929393854967847056065321e-84",
       "0.02501969643853000197268229"
      ],
      "c_g": [
       "-1.937902150132756234205634e-55",
       "-0.0005054866301181613849894282"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "-3.519892105428729074295261e-52",
       "2.741258741258741258741259"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-3.275591693335268045850584e-54",
       "0.02550999886701395886158583"
      ],
      "c_g": [
       "-2.388941748544467332349927e-56",
       "-0.0004902879861792438954539162"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "1.232452178978591316983793e-101",
       "3.080180112967654475804968"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.146913029995704886743185e-103",
       "0.02866398199096056752433741"
      ],
      "c_g": [
       "-4.756930844066413741280019e-55",
       "-0.0004816956068350579322539942"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     }
    ],
    "24": [
     {
      "s": [
       "-5.297038257226489638821057e-82",
       "2.68857172135926286741336"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-4.929393854967847056065321e-84",
       "0.02501969643853000197268229"
      ],
      "c_g": [
       "-2.023213256702529987828498e-55",
       "-0.0005172290729977108974950514"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "-3.519892105428729074295261e-52",
       "2.741258741258741258741259"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-3.275591693335268045850584e-54",
       "0.02550999886701395886158583"
      ],
      "c_g": [
       "-9.417740564627841048346329e-56",
       "-0.0005029900476979371166606276"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "1.232452178978591316983793e-101",
       "3.080180112967654475804968"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.146913029995704886743185e-103",
       "0.02866398199096056752433741"
      ],
      "c_g": [
       "-4.685511314596179160422582e-55",
       "-0.0004979792784536943451718762"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     }
    ],
    "32": [
     {
      "s": [
       "-5.297038257226489638821057e-82",
       "2.68857172135926286741336"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-4.929393854967847056065321e-84",
       "0.02501969643853000197268229"
      ],
      "c_g": [
       "-2.023213256702529987828498e-55",
       "-0.0005172290729977108974950514"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "-3.519892105428729074295261e-52",
       "2.741258741258741258741259"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-3.275591693335268045850584e-54",
       "0.02550999886701395886158583"
      ],
      "c_g": [
       "-9.417740564627841048346329e-56",
       "-0.0005029900476979371166606276"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "1.232452178978591316983793e-101",
       "3.080180112967654475804968"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.146913029995704886743185e-103",
       "0.02866398199096056752433741"
      ],
      "c_g": [
       "-4.685511314596179160422582e-55",
       "-0.0004979792784536943451718762"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     }
    ]
   },
   "Q": {
    "frac": "22/13",
    "value": 1.6923076923076923
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 2.7412587412587412
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 3.0801801129676543
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 2.688571721359263
     },
     "mult": 1
    }
   ],
   "representative": {
    "3.5": [
     0.5477225575051661,
     0.0
    ],
    "-1.5": [
     0.8366600265340756,
     0.0
    ]
   },
   "admission": "T5/R2"
  },
  "T1/R4": {
   "role": "row",
   "slot": "R4",
   "level": 6,
   "lam0": 48.0,
   "orbit": "v2",
   "sig": [
    2,
    0,
    8
   ],
   "k": 3,
   "r": 1,
   "points": [
    {
     "eps": 0.25,
     "read": false
    },
    {
     "eps": 0.3535533905932738,
     "read": false
    },
    {
     "eps": 0.5,
     "read": false
    },
    {
     "eps": 0.7071067811865476,
     "read": false
    },
    {
     "eps": 1.0,
     "read": false
    },
    {
     "eps": 1.4142135623730951,
     "read": true
    },
    {
     "eps": 2.0,
     "read": false
    },
    {
     "eps": 2.8284271247461903,
     "read": false
    },
    {
     "eps": 4.0,
     "read": false
    },
    {
     "eps": 5.656854249492381,
     "read": false
    },
    {
     "eps": 8.0,
     "read": false
    },
    {
     "eps": 11.313708498984761,
     "read": false
    },
    {
     "eps": 16.0,
     "read": false
    },
    {
     "eps": 22.627416997969522,
     "read": false
    },
    {
     "eps": 32.0,
     "read": true
    },
    {
     "eps": 45.254833995939045,
     "read": true
    },
    {
     "eps": 64.0,
     "read": true
    },
    {
     "eps": 72.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 4,
   "ranks": [
    1,
    0
   ],
   "chains": 3,
   "type": "Elliptic",
   "idx": [
    2,
    0,
    2
   ],
   "kr": 0,
   "kc": 0,
   "ki": 1,
   "next_order": {
    "16": [
     {
      "s": [
       "1.5894457303110833644197e-52",
       "0.04351204351204351204351204"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "2.7896762244219591265217e-54",
       "0.000763690832261409741414218"
      ],
      "c_g": [
       "7.526769180719597978705729e-58",
       "-0.00002244098854384017931369553"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.0",
       "0.7832167832167832167832168"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.01374643498070537534545592"
      ],
      "c_g": [
       "4.701576097919634429959845e-57",
       "-0.00003599772439608122331887315"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "0.0",
       "0.957264957264957264957265"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.0168011983097510143111128"
      ],
      "c_g": [
       "-3.246989148967665663728754e-55",
       "-0.0003051764629183947844546185"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "0.0",
       "1.305361305361305361305361"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.02291072496784229224242654"
      ],
      "c_g": [
       "5.957804799482451215134196e-58",
       "-0.000723591329962861222804008"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 4
     },
     {
      "s": [
       "0.0",
       "1.566433566433566433566434"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.02749286996141075069091185"
      ],
      "c_g": [
       "1.407422237170304511432112e-55",
       "-0.0004108756054299986547688377"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     }
    ],
    "24": [
     {
      "s": [
       "1.5894457303110833644197e-52",
       "0.04351204351204351204351204"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "2.7896762244219591265217e-54",
       "0.000763690832261409741414218"
      ],
      "c_g": [
       "7.526769180719597978705729e-58",
       "-0.000023443477364252014399328"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.0",
       "0.7832167832167832167832168"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.01374643498070537534545592"
      ],
      "c_g": [
       "7.280566465162657129361309e-57",
       "-0.00005725079872074393205200999"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "0.0",
       "0.957264957264957264957265"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.0168011983097510143111128"
      ],
      "c_g": [
       "-3.130018245755751970272971e-55",
       "-0.0003383410969539136454785312"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "0.0",
       "1.305361305361305361305361"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.02291072496784229224242654"
      ],
      "c_g": [
       "2.071210492358289816527159e-55",
       "-0.0007572385284926777063070296"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 4
     },
     {
      "s": [
       "0.0",
       "1.566433566433566433566434"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.02749286996141075069091185"
      ],
      "c_g": [
       "1.590910542030549394758898e-55",
       "-0.0004571083973030289374447451"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     }
    ],
    "32": [
     {
      "s": [
       "1.5894457303110833644197e-52",
       "0.04351204351204351204351204"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "2.7896762244219591265217e-54",
       "0.000763690832261409741414218"
      ],
      "c_g": [
       "7.526769180719597978705729e-58",
       "-0.000023443477364252014399328"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.0",
       "0.7832167832167832167832168"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.01374643498070537534545592"
      ],
      "c_g": [
       "7.280566465162657129361309e-57",
       "-0.00005725079872074393205200999"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "0.0",
       "0.957264957264957264957265"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.0168011983097510143111128"
      ],
      "c_g": [
       "-3.130018245755751970272971e-55",
       "-0.0003383410969539136454785312"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "0.0",
       "1.305361305361305361305361"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.02291072496784229224242654"
      ],
      "c_g": [
       "2.071210492358289816527159e-55",
       "-0.0007572385284926777063070296"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 4
     },
     {
      "s": [
       "0.0",
       "1.566433566433566433566434"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.02749286996141075069091185"
      ],
      "c_g": [
       "1.590910542030549394758898e-55",
       "-0.0004571083973030289374447451"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     }
    ]
   },
   "Q": {
    "frac": "147/143",
    "value": 1.027972027972028
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "frac": "56/1287",
      "value": 0.04351204351204351
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "112/143",
      "value": 0.7832167832167832
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "112/117",
      "value": 0.9572649572649573
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "224/143",
      "value": 1.5664335664335665
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "560/429",
      "value": 1.3053613053613053
     },
     "mult": 1
    }
   ],
   "representative": {
    "2": [
     1.0,
     0.0
    ]
   },
   "admission": "T1/R4"
  },
  "T4/R4": {
   "role": "row",
   "slot": "R4",
   "level": 6,
   "lam0": 48.0,
   "orbit": "outside3",
   "sig": [
    7,
    0,
    2
   ],
   "k": 4,
   "r": 2,
   "points": [
    {
     "eps": 0.25,
     "read": false
    },
    {
     "eps": 0.3535533905932738,
     "read": false
    },
    {
     "eps": 0.5,
     "read": false
    },
    {
     "eps": 0.7071067811865476,
     "read": false
    },
    {
     "eps": 1.0,
     "read": false
    },
    {
     "eps": 1.4142135623730951,
     "read": false
    },
    {
     "eps": 2.0,
     "read": false
    },
    {
     "eps": 2.8284271247461903,
     "read": false
    },
    {
     "eps": 4.0,
     "read": true
    },
    {
     "eps": 5.656854249492381,
     "read": false
    },
    {
     "eps": 8.0,
     "read": true
    },
    {
     "eps": 11.313708498984761,
     "read": true
    },
    {
     "eps": 16.0,
     "read": false
    },
    {
     "eps": 22.627416997969522,
     "read": false
    },
    {
     "eps": 32.0,
     "read": false
    },
    {
     "eps": 45.254833995939045,
     "read": false
    },
    {
     "eps": 64.0,
     "read": true
    },
    {
     "eps": 72.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 6,
   "ranks": [
    2,
    0
   ],
   "chains": 4,
   "type": "Elliptic",
   "idx": [
    7,
    1,
    6
   ],
   "kr": 0,
   "kc": 0,
   "ki": 3,
   "next_order": {
    "16": [
     {
      "s": [
       "-1.691791739876743108185433e-50",
       "0.005736859223094351705573823"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-2.575392243537360426429318e-52",
       "0.00008733145101240050550832012"
      ],
      "c_g": [
       "1.410053871390215170082864e-53",
       "-0.00001048206110242490919256803"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     },
     {
      "s": [
       "6.530848127863545995485519e-51",
       "0.1392518329643492375161716"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "9.941823934809887516860272e-53",
       "0.002119812279854689995527604"
      ],
      "c_g": [
       "-6.592836807179460173101994e-54",
       "0.00003714168119815899137041395"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "-2.421776100969072550449298e-51",
       "0.310692742361500122983186"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-3.686637804765676645711673e-53",
       "0.004729634623109431361833082"
      ],
      "c_g": [
       "-2.33994929613870534220547e-54",
       "-0.0003094939527389709984387031"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-1.942818970075171972213463e-51",
       "1.012091604894205959728147"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-2.957527683929569678677364e-53",
       "0.01540693696248758110710271"
      ],
      "c_g": [
       "1.288455257573316551381278e-54",
       "-0.00007626777955689850145987648"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     }
    ],
    "24": [
     {
      "s": [
       "-1.691791739876743108185433e-50",
       "0.005736859223094351705573823"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-2.575392243537360426429318e-52",
       "0.00008733145101240050550832012"
      ],
      "c_g": [
       "1.303155876373250699195963e-53",
       "-0.000009953348733660484493364564"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     },
     {
      "s": [
       "6.530848127863545995485519e-51",
       "0.1392518329643492375161716"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "9.941823934809887516860272e-53",
       "0.002119812279854689995527604"
      ],
      "c_g": [
       "-6.599756929252469066433429e-54",
       "0.00003211330512148688858971683"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "-2.421776100969072550449298e-51",
       "0.310692742361500122983186"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-3.686637804765676645711673e-53",
       "0.004729634623109431361833082"
      ],
      "c_g": [
       "-2.638450622803305377992096e-54",
       "-0.0003210698505576520471156753"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-1.942818970075171972213463e-51",
       "1.012091604894205959728147"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-2.957527683929569678677364e-53",
       "0.01540693696248758110710271"
      ],
      "c_g": [
       "1.307724233798739228682989e-54",
       "-0.0001235212562843326517621256"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     }
    ],
    "32": [
     {
      "s": [
       "-1.691791739876743108185433e-50",
       "0.005736859223094351705573823"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-2.575392243537360426429318e-52",
       "0.00008733145101240050550832012"
      ],
      "c_g": [
       "1.303155876373250699195963e-53",
       "-0.000009953348733660484493364564"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     },
     {
      "s": [
       "6.530848127863545995485519e-51",
       "0.1392518329643492375161716"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "9.941823934809887516860272e-53",
       "0.002119812279854689995527604"
      ],
      "c_g": [
       "-6.599756929252469066433429e-54",
       "0.00003211330512148688858971683"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "-2.421776100969072550449298e-51",
       "0.310692742361500122983186"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "-3.686637804765676645711673e-53",
       "0.004729634623109431361833082"
      ],
      "c_g": [
       "-2.638450622803305377992096e-54",
       "-0.0003210698505576520471156753"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "-1.942818970075171972213463e-51",
       "1.012091604894205959728147"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "-2.957527683929569678677364e-53",
       "0.01540693696248758110710271"
      ],
      "c_g": [
       "1.307724233798739228682989e-54",
       "-0.0001235212562843326517621256"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     }
    ]
   },
   "Q": {
    "value": 1.1852034613388465
   },
   "s": {
    "frac": "1/1",
    "value": 1.0
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "value": 1.01209160489
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 0.310692742362
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 0.139251832964
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "value": 0.00573685922309
     },
     "mult": 1
    }
   ],
   "representative": {
    "3": [
     -0.3086472568785893,
     -0.06949329411780472
    ],
    "2": [
     0.34579154134174556,
     -0.37948776525885436
    ],
    "1": [
     0.20302636427665244,
     0.1632131521070081
    ],
    "0": [
     0.29228686825253675,
     0.09882806648613832
    ],
    "-1": [
     0.3588169781640245,
     -0.4475251820209991
    ],
    "-2": [
     0.0005102297941505528,
     0.0441294932645358
    ],
    "-3": [
     0.22297893474867425,
     -0.30425691601898186
    ]
   },
   "admission": "T4/R4"
  },
  "T1/R5": {
   "role": "row",
   "slot": "R5",
   "level": 6,
   "lam0": 48.0,
   "orbit": "v2",
   "sig": [
    2,
    0,
    8
   ],
   "k": 3,
   "r": 1,
   "points": [
    {
     "eps": 0.25,
     "read": false
    },
    {
     "eps": 0.3535533905932738,
     "read": false
    },
    {
     "eps": 0.5,
     "read": false
    },
    {
     "eps": 0.7071067811865476,
     "read": false
    },
    {
     "eps": 1.0,
     "read": false
    },
    {
     "eps": 1.4142135623730951,
     "read": false
    },
    {
     "eps": 2.0,
     "read": false
    },
    {
     "eps": 2.8284271247461903,
     "read": false
    },
    {
     "eps": 4.0,
     "read": false
    },
    {
     "eps": 5.656854249492381,
     "read": false
    },
    {
     "eps": 8.0,
     "read": false
    },
    {
     "eps": 11.313708498984761,
     "read": false
    },
    {
     "eps": 16.0,
     "read": false
    },
    {
     "eps": 22.627416997969522,
     "read": false
    },
    {
     "eps": 32.0,
     "read": true
    }
   ],
   "cutoffs": [
    24,
    32,
    40
   ],
   "m815": null,
   "dim": 4,
   "ranks": [
    1,
    0
   ],
   "chains": 3,
   "type": "Elliptic",
   "idx": [
    2,
    0,
    2
   ],
   "kr": 0,
   "kc": 0,
   "ki": 1,
   "next_order": {
    "16": [
     {
      "s": [
       "8.33061766222011076041516e-54",
       "0.02447552447552447552447552"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.479743679047341547699521e-55",
       "0.0004347517087271278347207446"
      ],
      "c_g": [
       "2.112838696987356963002834e-58",
       "-0.00001406762205842468734374544"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.0",
       "0.4405594405594405594405594"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.007825530757088301024973402"
      ],
      "c_g": [
       "-1.078041010571423762627482e-56",
       "-0.0003214474837500513015169849"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "0.0",
       "0.5384615384615384615384615"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.00956453759199681236385638"
      ],
      "c_g": [
       "5.265496328392613607061761e-56",
       "-0.0003154919506442634452905777"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "0.0",
       "0.7342657342657342657342657"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.01304255126181383504162234"
      ],
      "c_g": [
       "-1.537644927568202268763357e-56",
       "-0.00008248680494604239833221933"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 4
     },
     {
      "s": [
       "0.0",
       "0.8811188811188811188811189"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.0156510615141766020499468"
      ],
      "c_g": [
       "-8.795857086759700701424846e-58",
       "-0.0002990434172606534149383853"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     }
    ],
    "24": [
     {
      "s": [
       "8.33061766222011076041516e-54",
       "0.02447552447552447552447552"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.479743679047341547699521e-55",
       "0.0004347517087271278347207446"
      ],
      "c_g": [
       "2.112838696987356963002834e-58",
       "-0.00001464217821196850767023877"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.0",
       "0.4405594405594405594405594"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.007825530757088301024973402"
      ],
      "c_g": [
       "3.824545785679527502762826e-56",
       "-0.000333601589230575717098612"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "0.0",
       "0.5384615384615384615384615"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.00956453759199681236385638"
      ],
      "c_g": [
       "2.957930514714808627260293e-56",
       "-0.0003344072563731941746387781"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "0.0",
       "0.7342657342657342657342657"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.01304255126181383504162234"
      ],
      "c_g": [
       "-2.001264723542928123289454e-56",
       "-0.0001017413242805997621650564"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 4
     },
     {
      "s": [
       "0.0",
       "0.8811188811188811188811189"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.0156510615141766020499468"
      ],
      "c_g": [
       "-1.126971965725722973961297e-56",
       "-0.0003254565068009672365883763"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     }
    ],
    "32": [
     {
      "s": [
       "8.33061766222011076041516e-54",
       "0.02447552447552447552447552"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "-",
      "kappa_s": [
       "1.479743679047341547699521e-55",
       "0.0004347517087271278347207446"
      ],
      "c_g": [
       "2.112838696987356963002834e-58",
       "-0.00001464217821196850767023877"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 0
     },
     {
      "s": [
       "0.0",
       "0.4405594405594405594405594"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.007825530757088301024973402"
      ],
      "c_g": [
       "3.824545785679527502762826e-56",
       "-0.000333601589230575717098612"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 1
     },
     {
      "s": [
       "0.0",
       "0.5384615384615384615384615"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.00956453759199681236385638"
      ],
      "c_g": [
       "2.957930514714808627260293e-56",
       "-0.0003344072563731941746387781"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 2
     },
     {
      "s": [
       "0.0",
       "0.7342657342657342657342657"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.01304255126181383504162234"
      ],
      "c_g": [
       "-2.001264723542928123289454e-56",
       "-0.0001017413242805997621650564"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 4
     },
     {
      "s": [
       "0.0",
       "0.8811188811188811188811189"
      ],
      "mult": 1,
      "cls": "imag",
      "krein": "+",
      "kappa_s": [
       "0.0",
       "0.0156510615141766020499468"
      ],
      "c_g": [
       "-1.126971965725722973961297e-56",
       "-0.0003254565068009672365883763"
      ],
      "c_g_off": null,
      "under_bar": true,
      "spectrum_index": 3
     }
    ]
   },
   "Q": {
    "frac": "581/572",
    "value": 1.0157342657342658
   },
   "s": {
    "frac": "9/16",
    "value": 0.5625
   },
   "spectrum": [
    {
     "cls": "imag",
     "sign": -1,
     "tau": {
      "frac": "56/1287",
      "value": 0.04351204351204351
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "112/143",
      "value": 0.7832167832167832
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "112/117",
      "value": 0.9572649572649573
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "224/143",
      "value": 1.5664335664335665
     },
     "mult": 1
    },
    {
     "cls": "imag",
     "sign": 1,
     "tau": {
      "frac": "560/429",
      "value": 1.3053613053613053
     },
     "mult": 1
    }
   ],
   "representative": {
    "2": [
     1.0,
     0.0
    ]
   },
   "admission": "T1/R5"
  }
 },
 "census": {
  "arm_a": {
   "cutoffs": [
    24,
    32,
    40
   ],
   "deciding": [
    32,
    40
   ],
   "pairs": [
    [
     24,
     32
    ],
    [
     32,
     40
    ]
   ]
  },
  "arm_b": {
   "cutoffs": [
    16,
    24,
    32
   ],
   "deciding": [
    24,
    32
   ],
   "pairs": [
    [
     24,
     32
    ]
   ]
  },
  "top": 40,
  "band_n": 16,
  "beside_n": 24,
  "band_level": {
   "R2": 17,
   "R4": 18,
   "R5": 18
  },
  "windows": {
   "R4": [
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
    5.656854249492381,
    8.0,
    11.313708498984761,
    16.0,
    22.627416997969522,
    32.0,
    45.254833995939045,
    64.0,
    72.0
   ],
   "R5": [
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
    5.656854249492381,
    8.0,
    11.313708498984761,
    16.0,
    22.627416997969522,
    32.0
   ],
   "R2": [
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
    5.656854249492381,
    8.0,
    11.313708498984761,
    16.0,
    22.627416997969522,
    32.0,
    45.254833995939045,
    64.0,
    90.50966799187809,
    128.0,
    132.0
   ]
  },
  "rows": [
   "T1/R4",
   "T4/R4",
   "T1/R5",
   "T4/R5",
   "T5/R2"
  ],
  "targets": [
   "T5/R2",
   "T1/R4",
   "T4/R4",
   "T1/R5"
  ],
  "controls": [
   "C1",
   "C2a",
   "C2b",
   "C3",
   "C2a/R5",
   "P5",
   "C3R/R4"
  ],
  "read_controls": [
   "C2a",
   "C2b",
   "C3",
   "C2a/R5",
   "P5",
   "C3R/R4"
  ],
  "run_order": [
   "T4/R5",
   "C1",
   "C2a",
   "C2b",
   "C3",
   "C2a/R5",
   "P5",
   "C3R/R4",
   "T5/R2",
   "T1/R4",
   "T4/R4",
   "T1/R5"
  ],
  "twelve": {
   "T1/R4": [
    1.4142135623730951,
    32.0,
    45.254833995939045,
    64.0,
    72.0
   ],
   "T4/R4": [
    4.0,
    8.0,
    11.313708498984761,
    64.0,
    72.0
   ],
   "T1/R5": [
    32.0
   ],
   "T5/R2": [
    45.254833995939045
   ]
  },
  "paths": {
   "T1/R4": [
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
    5.656854249492381,
    8.0,
    11.313708498984761,
    16.0,
    22.627416997969522,
    32.0,
    45.254833995939045,
    64.0,
    72.0
   ],
   "T4/R4": [
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
    5.656854249492381,
    8.0,
    11.313708498984761,
    16.0,
    22.627416997969522,
    32.0,
    45.254833995939045,
    64.0,
    72.0
   ],
   "T1/R5": [
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
    5.656854249492381,
    8.0,
    11.313708498984761,
    16.0,
    22.627416997969522,
    32.0
   ],
   "T5/R2": [
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
    5.656854249492381,
    8.0,
    11.313708498984761,
    16.0,
    22.627416997969522,
    32.0,
    45.254833995939045
   ],
   "T4/R5": [
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
    5.656854249492381,
    8.0,
    11.313708498984761
   ]
  },
  "bracket_b": [
   8.0,
   11.313708498984761
  ],
  "moved": 0.4,
  "attempts": 3,
  "w": 0.001,
  "w_half": 0.0005,
  "deltas": [
   0.01,
   0.05
  ],
  "agree": 0.01
 },
 "inherited": {
  "values": {
   "calibration": {
    "kappa": 2.0,
    "m0": 0.2,
    "G": 3.0,
    "kappa_d": 10.0,
    "kappa_L": 200.0,
    "leak_cap": 0.0001,
    "L_cut_over_u": 1000.0,
    "G_L": 10.0,
    "D_cut": 1e-08,
    "u": 2.220446049250313e-16
   },
   "solve": {
    "accept_rel": 1e-15,
    "halving": 1e-08,
    "secant": 1e-06,
    "secant_iterations": 8,
    "seed_check": 1e-12
   },
   "checks": {
    "bar": 0.05,
    "overlap_xy": 0.9,
    "overlap_z": 0.1,
    "identity": 0.01,
    "axis": 1e-06,
    "factor": 0.1,
    "span": 1e-06,
    "left": 10000.0,
    "conditioning": 0.01,
    "idempotence": 1e-06,
    "slow_band": 2,
    "band_edge": 0.5,
    "orthonormal": 1e-06,
    "k6_eps": 2
   },
   "next_order": {
    "points": [
     0.25,
     0.5,
     1.0
    ],
    "ceiling": [
     0.1,
     0.001
    ],
    "resolution": 0.1,
    "floor": 0.1,
    "nominal_weights": [
     42.666666666666664,
     8.0,
     0.3333333333333333
    ]
   },
   "cutoffs": [
    16,
    24,
    32
   ],
   "c1": {
    "eps": [
     0.5,
     1.0,
     2.0
    ],
    "cutoffs": [
     16,
     24,
     32,
     40
    ],
    "residual": 1e-15,
    "off_level": 1e-14,
    "spectrum": 1e-10,
    "along": 1e-12
   },
   "sample": [
    0.25,
    0.5,
    1.0,
    2.0
   ]
  },
  "next_order": {
   "file": "run/next_order_values.json",
   "sha256": "b05fff5ddd71c29c79318b6257f940c6035d7c954f3bafa34a22080eecc00333",
   "cases_sha256": "45ad237179b6c7176b4372f8b36fe6cf20ba95ebdff510f727eec1ea4a2ebeea",
   "script": {
    "path": "run/next_order_values.py",
    "sha256": "4bf81a60beef1338c6da36b2d4c04f9d51b826322a0b1e9540769fa318f4028e"
   },
   "environment": {
    "python": "3.13.13",
    "sympy": "1.14.0",
    "mpmath": "1.3.0",
    "numpy": "2.5.0",
    "dps": 50
   }
  },
  "terms_sha256": "9d16712982e5388970fca33de59958721109e09f9d88dec7a2a088489ad3efd6",
  "manifest_sha256": "8cc2d7e2d01a72fc3572adabf917656abf36cbddd70ed35532b73ae5451790dd",
  "go": "https://github.com/openwave-labs/openwave/pull/650#issuecomment-6385959498"
 },
 "terms_sha256": "924e7ee363a756a88ba2a9ee9bc03eb56c4fe27778c6e8f920a42a5229034a4c",
 "manifest_sha256": "6d244f4af56b71d60f17077724628b7489b278e84a7d24565c17038ed19ba43a",
 "go": "https://github.com/openwave-labs/openwave/pull/687#issuecomment-6548866246",
 "targets": [
  {
   "name": "reference/m8_16_1/run/out/RUNLOG.json",
   "file": "reference/m8_16_1/run/out/RUNLOG.json",
   "sha256": "84f2c9f6667e02d030dfe1d0f759ad183cd0bc6293228bd60e3aa0a2785edadb",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/out/RUNLOG.json",
   "blob": "ccdd58fe8a4498b1d94ebf272ce2c80aaafb193e"
  },
  {
   "name": "reference/m8_16_1/run/out/C1.json",
   "file": "reference/m8_16_1/run/out/C1.json",
   "sha256": "356ffbf4fe71eb1429163e849d53a33f92643b5784124a9d11e9ef8c5e9b15ca",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/out/C1.json",
   "blob": "4ccac3d2843c4082fb13af7d34ece7b49c15fe07"
  },
  {
   "name": "reference/m8_16_1/run/out/C2a.json",
   "file": "reference/m8_16_1/run/out/C2a.json",
   "sha256": "0165455b6ba66d462888521a5273fc8f0a744d4430c808e2855f5e9d06d3c24f",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/out/C2a.json",
   "blob": "91564933c228cdc5df300decaae4e62602d42b4b"
  },
  {
   "name": "reference/m8_16_1/run/out/C2b.json",
   "file": "reference/m8_16_1/run/out/C2b.json",
   "sha256": "efd5c21e233f08897f5dc9ded449801bfa3fe8d5e8895b6099b1840dd2b5a0fd",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/out/C2b.json",
   "blob": "19f80bada697e9d7e3a9274270089cb2e0c8cc1c"
  },
  {
   "name": "reference/m8_16_1/run/out/C3.json",
   "file": "reference/m8_16_1/run/out/C3.json",
   "sha256": "8e7c303c2e7dfe35e39e3f581ace30d537a1f30073389f8e6aa199b30250d469",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/out/C3.json",
   "blob": "8ebe57cbb36e2e9704f7bd6208901311e5b82a70"
  },
  {
   "name": "reference/m8_16_1/run/out/C2a_R5.json",
   "file": "reference/m8_16_1/run/out/C2a_R5.json",
   "sha256": "387ed7b8e04030a2e10569eedb313ca5713bea92bec5a6bac7bd2e6697e668be",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/out/C2a_R5.json",
   "blob": "af6c5fce0d2befbbade54a375c7db89e391226a2"
  },
  {
   "name": "reference/m8_16_1/run/out/P5.json",
   "file": "reference/m8_16_1/run/out/P5.json",
   "sha256": "c404e49c2a6e52b0754c97349b01c8097db3271af641bc56071fd72098082bd0",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/out/P5.json",
   "blob": "b123611e0b013fadf9d449ee90302f6924ebbf6a"
  },
  {
   "name": "reference/m8_16_1/run/out/C3R_R4.json",
   "file": "reference/m8_16_1/run/out/C3R_R4.json",
   "sha256": "21706fd5b2548cdcb0bf8b44d2df0c6b62d34abad41363d0fb1e43a14a5c9fb6",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/out/C3R_R4.json",
   "blob": "dfc8566b1f1969030be5b5161f17b83fdd16e2f5"
  },
  {
   "name": "reference/m8_16_1/run/FROZEN.json",
   "file": "reference/m8_16_1/run/FROZEN.json",
   "sha256": "c258e299b8f0268e805712b4f087b3533c6e0cd3c37e237bf3d0639540c52a2d",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/FROZEN.json",
   "blob": "9da1269b789bc8703d0ec819657b70bcffae1f9c"
  },
  {
   "name": "reference/m8_16_1/run/m8_16_task_details.md",
   "file": "reference/m8_16_1/run/m8_16_task_details.md",
   "sha256": "9d16712982e5388970fca33de59958721109e09f9d88dec7a2a088489ad3efd6",
   "commit": "292b158cf1a53902489ced3426e1f047699aeb62",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16-1/run/m8_16_task_details.md",
   "blob": "746fb0c38ceb994caf9f32096d455e63c1f4a80b"
  },
  {
   "name": "reference/m8_17/run/out/RUNLOG.json",
   "file": "reference/m8_17/run/out/RUNLOG.json",
   "sha256": "7d761506f8bfb05cee50ddc6cbf29b0a97fb20c7d9242d57c6765cff8114196a",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/out/RUNLOG.json",
   "blob": "d3e29ce7d6f5ce64b721e17b188c72db2e174b64"
  },
  {
   "name": "reference/m8_17/run/out/T1_R4.json",
   "file": "reference/m8_17/run/out/T1_R4.json",
   "sha256": "06e51a090e310a2accf6bab21fdee693ab2806ff6115bb8a7f10b54aae96bcc9",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/out/T1_R4.json",
   "blob": "d7480d323c91df14a990aea1cfe3a7f4751d53f9"
  },
  {
   "name": "reference/m8_17/run/out/T1_R5.json",
   "file": "reference/m8_17/run/out/T1_R5.json",
   "sha256": "e8ecd65f3754e6bd047e0d6c34964d59aaa44bd05bb522a71a884118926d3678",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/out/T1_R5.json",
   "blob": "1740687e6215cc0783defdb279db939ef3a3cf95"
  },
  {
   "name": "reference/m8_17/run/out/T4_R4.json",
   "file": "reference/m8_17/run/out/T4_R4.json",
   "sha256": "2955bc02cb2c653deb28f13bcdf53229b21541521681900661183a632823e512",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/out/T4_R4.json",
   "blob": "2e825c81b3e8dcb52daf25fbfd0f4b09a80f713c"
  },
  {
   "name": "reference/m8_17/run/out/T4_R5.json",
   "file": "reference/m8_17/run/out/T4_R5.json",
   "sha256": "167ddab0d696d2758581a8f7063d0d74283c13bbe2500fb60c91665e1a2ce95c",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/out/T4_R5.json",
   "blob": "1cb6cee89f8dc653f038e6c894336ceb9349d656"
  },
  {
   "name": "reference/m8_17/run/out/T5_R2.json",
   "file": "reference/m8_17/run/out/T5_R2.json",
   "sha256": "fb185a91edcf7099950e4e7529d125136537769992d8bddb9cb18edf0d632bec",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/out/T5_R2.json",
   "blob": "5501e2c7590788242ba5da7bb8d07cf2710bc72c"
  },
  {
   "name": "reference/m8_17/run/out/locate_T4_R5_8_11.313708499.json",
   "file": "reference/m8_17/run/out/locate_T4_R5_8_11.313708499.json",
   "sha256": "517aa9f62a45255ec4a76690e95069935d32265e532a7858e58d8709cf10af55",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/out/locate_T4_R5_8_11.313708499.json",
   "blob": "cee35076c3670febf1f4e08f9baeea571d89ff20"
  },
  {
   "name": "reference/m8_17/run/out/CENSUS.json",
   "file": "reference/m8_17/run/out/CENSUS.json",
   "sha256": "d010db400a6bdef2ad64ce1bfd0e085995c49e62ba1982325ff6234f8d233aac",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/out/CENSUS.json",
   "blob": "626c25386b93a36f9b7c5bd9dd5215e1822c05f9"
  },
  {
   "name": "reference/m8_17/run/census.py",
   "file": "reference/m8_17/run/census.py",
   "sha256": "f4c60943215bea341656638ec431ee79206d0561fd266906b018196cfdb5421d",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/census.py",
   "blob": "7be181f99c814cd422ffdb34d80947fa9a9aa5c0"
  },
  {
   "name": "reference/m8_17/run/census_lib.py",
   "file": "reference/m8_17/run/census_lib.py",
   "sha256": "4873006e94e09bdf1b03396e2e9e85881ce503d2def0a2036858691706d6a321",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/census_lib.py",
   "blob": "2889ff860096c8e610115bdc39a085e5e8619fac"
  },
  {
   "name": "reference/m8_17/run/m8_16_task_details.md",
   "file": "reference/m8_17/run/m8_16_task_details.md",
   "sha256": "667159c3db993396a10a75775acf79e78f744d161c641ebc37c9e6baf3af7cea",
   "commit": "7b66f8a0e207888042e6bb4a9ab636f946cb2a81",
   "path": "files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/run/m8_16_task_details.md",
   "blob": "c6b2b13a88b59734ac949d18541f5490ea49332d"
  },
  {
   "name": "SOURCES.json",
   "file": "reference/SOURCES.json",
   "sha256": "c90e734e8817423ed8053475072dc44efa2fc76db7cec974e1321be421a4da31"
  }
 ],
 "written": "2026-10-31T21:13:28Z"
}
```

</details>

## The order

1. T4/R5
2. C1
3. C2a
4. C2b
5. C3
6. C2a/R5
7. P5
8. C3R/R4
9. T5/R2
10. T1/R4
11. T4/R4
12. T1/R5
