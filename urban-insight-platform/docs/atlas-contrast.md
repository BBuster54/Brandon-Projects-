# Urban Atlas contrast audit

Computed WCAG sRGB relative-luminance ratios. Raw EPA fills are diagnostic, not standalone accessible boundaries. A 2px inner black/white keyline contrasts against the fill; a separate outer black (light) / white (dark) boundary contrasts against the surface or filtered tile. Category names remain text. Focus rings use a surface gap. Disabled filled buttons keep their contrast and use a dashed boundary, not opacity.

Light tile filter: grayscale(1) sepia(.3) brightness(.8) contrast(.4) brightness(1.3), all channels bounded by 0.39–0.806 (rounded #636363–#CECECE). Dark filter: grayscale(1) sepia(.25) invert(1) contrast(.3) brightness(.45), bounded by 0.1575–0.2925 (#282828–#4B4B4B). The filter applies only to .leaflet-tile-pane; UI/attribution/markers are not filtered. Bounds are conservative channel extrema, independent of original tile colors.

| Theme | Pairing | Foreground | Background | Ratio | Minimum | Note |
|---|---|---|---|---:|---:|---|
| light | paper / paper | #E5D7BB | #E5D7BB | 1.00 | — | Background/context only; not a text pairing |
| light | paper / surface | #E5D7BB | #FFF7E9 | 1.34 | — | Background/context only; not a text pairing |
| light | surface / paper | #FFF7E9 | #E5D7BB | 1.34 | — | Background/context only; not a text pairing |
| light | surface / surface | #FFF7E9 | #FFF7E9 | 1.00 | — | Background/context only; not a text pairing |
| light | ink / paper | #2B251C | #E5D7BB | 10.67 | 5.7 | Intended text or boundary pairing |
| light | ink / surface | #2B251C | #FFF7E9 | 14.26 | 5.7 | Intended text or boundary pairing |
| light | muted / paper | #534B3F | #E5D7BB | 6.04 | 5.7 | Intended text or boundary pairing |
| light | muted / surface | #534B3F | #FFF7E9 | 8.07 | 5.7 | Intended text or boundary pairing |
| light | accent / paper | #4B4232 | #E5D7BB | 6.95 | 5.7 | Intended text or boundary pairing |
| light | accent / surface | #4B4232 | #FFF7E9 | 9.29 | 5.7 | Intended text or boundary pairing |
| light | control / paper | #6A5E4C | #E5D7BB | 4.45 | 4.28 | Intended text or boundary pairing |
| light | control / surface | #6A5E4C | #FFF7E9 | 5.95 | 4.28 | Intended text or boundary pairing |
| light | buttonText / paper | #FFF7E9 | #E5D7BB | 1.34 | — | Background/context only; not a text pairing |
| light | buttonText / surface | #FFF7E9 | #FFF7E9 | 1.00 | — | Background/context only; not a text pairing |
| light | edge / paper | #000000 | #E5D7BB | 14.77 | 3 | Intended text or boundary pairing |
| light | edge / surface | #000000 | #FFF7E9 | 19.73 | 3 | Intended text or boundary pairing |
| light | Filled button normal: text / fill | #FFF7E9 | #4B4232 | 9.29 | 5.7 |  |
| light | Filled button normal: boundary / paper | #4B4232 | #E5D7BB | 6.95 | 3 |  |
| light | Filled button normal: boundary / surface | #4B4232 | #FFF7E9 | 9.29 | 3 |  |
| light | Filled button hover: text / fill | #FFF7E9 | #2B251C | 14.26 | 5.7 |  |
| light | Filled button hover: boundary / paper | #2B251C | #E5D7BB | 10.67 | 3 |  |
| light | Filled button hover: boundary / surface | #2B251C | #FFF7E9 | 14.26 | 3 |  |
| light | Filled button disabled: text / fill | #FFF7E9 | #4B4232 | 9.29 | 5.7 |  |
| light | Filled button disabled: boundary / paper | #4B4232 | #E5D7BB | 6.95 | 3 |  |
| light | Filled button disabled: boundary / surface | #4B4232 | #FFF7E9 | 9.29 | 3 |  |
| light | Filled button focus: text / fill | #FFF7E9 | #4B4232 | 9.29 | 5.7 |  |
| light | Filled button focus: boundary / paper | #4B4232 | #E5D7BB | 6.95 | 3 |  |
| light | Filled button focus: boundary / surface | #4B4232 | #FFF7E9 | 9.29 | 3 |  |
| light | Focus ring / surface | #2B251C | #FFF7E9 | 14.26 | 3 |  |
| light | Focus ring / page | #2B251C | #E5D7BB | 10.67 | 3 |  |
| light | Inverted card text / fill | #E5D7BB | #2B251C | 10.67 | 5.7 |  |
| light | Good: raw fill / paper | #00e400 | #E5D7BB | 1.22 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Good: raw fill / surface | #00e400 | #FFF7E9 | 1.63 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Good: inner keyline / fill | #000000 | #00e400 | 12.10 | 3 |  |
| light | Good: outer keyline / paper | #000000 | #E5D7BB | 14.77 | 3 |  |
| light | Good: outer keyline / surface | #000000 | #FFF7E9 | 19.73 | 3 |  |
| light | Good: map outer boundary / filtered tile min | #000000 | #636363 | 3.50 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Good: map outer boundary / filtered tile max | #000000 | #cecece | 13.34 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Moderate: raw fill / paper | #ffff00 | #E5D7BB | 1.32 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Moderate: raw fill / surface | #ffff00 | #FFF7E9 | 1.01 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Moderate: inner keyline / fill | #000000 | #ffff00 | 19.56 | 3 |  |
| light | Moderate: outer keyline / paper | #000000 | #E5D7BB | 14.77 | 3 |  |
| light | Moderate: outer keyline / surface | #000000 | #FFF7E9 | 19.73 | 3 |  |
| light | Moderate: map outer boundary / filtered tile min | #000000 | #636363 | 3.50 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Moderate: map outer boundary / filtered tile max | #000000 | #cecece | 13.34 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Unhealthy for sensitive groups: raw fill / paper | #ff7e00 | #E5D7BB | 1.79 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Unhealthy for sensitive groups: raw fill / surface | #ff7e00 | #FFF7E9 | 2.40 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Unhealthy for sensitive groups: inner keyline / fill | #000000 | #ff7e00 | 8.24 | 3 |  |
| light | Unhealthy for sensitive groups: outer keyline / paper | #000000 | #E5D7BB | 14.77 | 3 |  |
| light | Unhealthy for sensitive groups: outer keyline / surface | #000000 | #FFF7E9 | 19.73 | 3 |  |
| light | Unhealthy for sensitive groups: map outer boundary / filtered tile min | #000000 | #636363 | 3.50 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Unhealthy for sensitive groups: map outer boundary / filtered tile max | #000000 | #cecece | 13.34 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Unhealthy: raw fill / paper | #ff0000 | #E5D7BB | 2.81 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Unhealthy: raw fill / surface | #ff0000 | #FFF7E9 | 3.76 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Unhealthy: inner keyline / fill | #000000 | #ff0000 | 5.25 | 3 |  |
| light | Unhealthy: outer keyline / paper | #000000 | #E5D7BB | 14.77 | 3 |  |
| light | Unhealthy: outer keyline / surface | #000000 | #FFF7E9 | 19.73 | 3 |  |
| light | Unhealthy: map outer boundary / filtered tile min | #000000 | #636363 | 3.50 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Unhealthy: map outer boundary / filtered tile max | #000000 | #cecece | 13.34 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Very unhealthy: raw fill / paper | #8f3f97 | #E5D7BB | 4.44 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Very unhealthy: raw fill / surface | #8f3f97 | #FFF7E9 | 5.93 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Very unhealthy: inner keyline / fill | #ffffff | #8f3f97 | 6.31 | 3 |  |
| light | Very unhealthy: outer keyline / paper | #000000 | #E5D7BB | 14.77 | 3 |  |
| light | Very unhealthy: outer keyline / surface | #000000 | #FFF7E9 | 19.73 | 3 |  |
| light | Very unhealthy: map outer boundary / filtered tile min | #000000 | #636363 | 3.50 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Very unhealthy: map outer boundary / filtered tile max | #000000 | #cecece | 13.34 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Hazardous: raw fill / paper | #7e0023 | #E5D7BB | 7.73 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Hazardous: raw fill / surface | #7e0023 | #FFF7E9 | 10.32 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| light | Hazardous: inner keyline / fill | #ffffff | #7e0023 | 10.99 | 3 |  |
| light | Hazardous: outer keyline / paper | #000000 | #E5D7BB | 14.77 | 3 |  |
| light | Hazardous: outer keyline / surface | #000000 | #FFF7E9 | 19.73 | 3 |  |
| light | Hazardous: map outer boundary / filtered tile min | #000000 | #636363 | 3.50 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| light | Hazardous: map outer boundary / filtered tile max | #000000 | #cecece | 13.34 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | paper / paper | #211E18 | #211E18 | 1.00 | — | Background/context only; not a text pairing |
| dark | paper / surface | #211E18 | #2D281F | 1.14 | — | Background/context only; not a text pairing |
| dark | surface / paper | #2D281F | #211E18 | 1.14 | — | Background/context only; not a text pairing |
| dark | surface / surface | #2D281F | #2D281F | 1.00 | — | Background/context only; not a text pairing |
| dark | ink / paper | #F7EAD0 | #211E18 | 13.96 | 8.57 | Intended text or boundary pairing |
| dark | ink / surface | #F7EAD0 | #2D281F | 12.29 | 8.57 | Intended text or boundary pairing |
| dark | muted / paper | #D8C7A7 | #211E18 | 10.01 | 8.57 | Intended text or boundary pairing |
| dark | muted / surface | #D8C7A7 | #2D281F | 8.81 | 8.57 | Intended text or boundary pairing |
| dark | accent / paper | #E4D0AC | #211E18 | 11.01 | 8.57 | Intended text or boundary pairing |
| dark | accent / surface | #E4D0AC | #2D281F | 9.70 | 8.57 | Intended text or boundary pairing |
| dark | control / paper | #B4A184 | #211E18 | 6.63 | 5.44 | Intended text or boundary pairing |
| dark | control / surface | #B4A184 | #2D281F | 5.83 | 5.44 | Intended text or boundary pairing |
| dark | buttonText / paper | #211E18 | #211E18 | 1.00 | — | Background/context only; not a text pairing |
| dark | buttonText / surface | #211E18 | #2D281F | 1.14 | — | Background/context only; not a text pairing |
| dark | edge / paper | #FFFFFF | #211E18 | 16.62 | 3 | Intended text or boundary pairing |
| dark | edge / surface | #FFFFFF | #2D281F | 14.64 | 3 | Intended text or boundary pairing |
| dark | Filled button normal: text / fill | #211E18 | #E4D0AC | 11.01 | 8.57 |  |
| dark | Filled button normal: boundary / paper | #E4D0AC | #211E18 | 11.01 | 3 |  |
| dark | Filled button normal: boundary / surface | #E4D0AC | #2D281F | 9.70 | 3 |  |
| dark | Filled button hover: text / fill | #211E18 | #F7EAD0 | 13.96 | 8.57 |  |
| dark | Filled button hover: boundary / paper | #F7EAD0 | #211E18 | 13.96 | 3 |  |
| dark | Filled button hover: boundary / surface | #F7EAD0 | #2D281F | 12.29 | 3 |  |
| dark | Filled button disabled: text / fill | #211E18 | #E4D0AC | 11.01 | 8.57 |  |
| dark | Filled button disabled: boundary / paper | #E4D0AC | #211E18 | 11.01 | 3 |  |
| dark | Filled button disabled: boundary / surface | #E4D0AC | #2D281F | 9.70 | 3 |  |
| dark | Filled button focus: text / fill | #211E18 | #E4D0AC | 11.01 | 8.57 |  |
| dark | Filled button focus: boundary / paper | #E4D0AC | #211E18 | 11.01 | 3 |  |
| dark | Filled button focus: boundary / surface | #E4D0AC | #2D281F | 9.70 | 3 |  |
| dark | Focus ring / surface | #F7EAD0 | #2D281F | 12.29 | 3 |  |
| dark | Focus ring / page | #F7EAD0 | #211E18 | 13.96 | 3 |  |
| dark | Inverted card text / fill | #211E18 | #F7EAD0 | 13.96 | 8.57 |  |
| dark | Good: raw fill / paper | #00e400 | #211E18 | 9.57 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Good: raw fill / surface | #00e400 | #2D281F | 8.43 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Good: inner keyline / fill | #000000 | #00e400 | 12.10 | 3 |  |
| dark | Good: outer keyline / paper | #FFFFFF | #211E18 | 16.62 | 3 |  |
| dark | Good: outer keyline / surface | #FFFFFF | #2D281F | 14.64 | 3 |  |
| dark | Good: map outer boundary / filtered tile min | #FFFFFF | #282828 | 14.74 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Good: map outer boundary / filtered tile max | #FFFFFF | #4b4b4b | 8.72 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Moderate: raw fill / paper | #ffff00 | #211E18 | 15.48 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Moderate: raw fill / surface | #ffff00 | #2D281F | 13.63 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Moderate: inner keyline / fill | #000000 | #ffff00 | 19.56 | 3 |  |
| dark | Moderate: outer keyline / paper | #FFFFFF | #211E18 | 16.62 | 3 |  |
| dark | Moderate: outer keyline / surface | #FFFFFF | #2D281F | 14.64 | 3 |  |
| dark | Moderate: map outer boundary / filtered tile min | #FFFFFF | #282828 | 14.74 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Moderate: map outer boundary / filtered tile max | #FFFFFF | #4b4b4b | 8.72 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Unhealthy for sensitive groups: raw fill / paper | #ff7e00 | #211E18 | 6.52 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Unhealthy for sensitive groups: raw fill / surface | #ff7e00 | #2D281F | 5.74 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Unhealthy for sensitive groups: inner keyline / fill | #000000 | #ff7e00 | 8.24 | 3 |  |
| dark | Unhealthy for sensitive groups: outer keyline / paper | #FFFFFF | #211E18 | 16.62 | 3 |  |
| dark | Unhealthy for sensitive groups: outer keyline / surface | #FFFFFF | #2D281F | 14.64 | 3 |  |
| dark | Unhealthy for sensitive groups: map outer boundary / filtered tile min | #FFFFFF | #282828 | 14.74 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Unhealthy for sensitive groups: map outer boundary / filtered tile max | #FFFFFF | #4b4b4b | 8.72 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Unhealthy: raw fill / paper | #ff0000 | #211E18 | 4.16 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Unhealthy: raw fill / surface | #ff0000 | #2D281F | 3.66 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Unhealthy: inner keyline / fill | #000000 | #ff0000 | 5.25 | 3 |  |
| dark | Unhealthy: outer keyline / paper | #FFFFFF | #211E18 | 16.62 | 3 |  |
| dark | Unhealthy: outer keyline / surface | #FFFFFF | #2D281F | 14.64 | 3 |  |
| dark | Unhealthy: map outer boundary / filtered tile min | #FFFFFF | #282828 | 14.74 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Unhealthy: map outer boundary / filtered tile max | #FFFFFF | #4b4b4b | 8.72 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Very unhealthy: raw fill / paper | #8f3f97 | #211E18 | 2.63 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Very unhealthy: raw fill / surface | #8f3f97 | #2D281F | 2.32 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Very unhealthy: inner keyline / fill | #ffffff | #8f3f97 | 6.31 | 3 |  |
| dark | Very unhealthy: outer keyline / paper | #FFFFFF | #211E18 | 16.62 | 3 |  |
| dark | Very unhealthy: outer keyline / surface | #FFFFFF | #2D281F | 14.64 | 3 |  |
| dark | Very unhealthy: map outer boundary / filtered tile min | #FFFFFF | #282828 | 14.74 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Very unhealthy: map outer boundary / filtered tile max | #FFFFFF | #4b4b4b | 8.72 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Hazardous: raw fill / paper | #7e0023 | #211E18 | 1.51 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Hazardous: raw fill / surface | #7e0023 | #2D281F | 1.33 | — | Diagnostic: fill alone may fail; not the boundary treatment |
| dark | Hazardous: inner keyline / fill | #ffffff | #7e0023 | 10.99 | 3 |  |
| dark | Hazardous: outer keyline / paper | #FFFFFF | #211E18 | 16.62 | 3 |  |
| dark | Hazardous: outer keyline / surface | #FFFFFF | #2D281F | 14.64 | 3 |  |
| dark | Hazardous: map outer boundary / filtered tile min | #FFFFFF | #282828 | 14.74 | 3 | Tile-filter output bound, covers every original sRGB pixel |
| dark | Hazardous: map outer boundary / filtered tile max | #FFFFFF | #4b4b4b | 8.72 | 3 | Tile-filter output bound, covers every original sRGB pixel |
