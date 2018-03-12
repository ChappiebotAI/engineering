## Chappie Computer Vison APIs

HTTP/1.1

Host URL: `cv.chappiebot.com`

Header:
    X-Api-Key: ...

Method: POST

### Labeling an image

POST https://cv.chappiebot.com/v2/labeling

Params:
```
    file: attacted in post request with format jpg/png, require minimum size 200x100 pixel and maximum 10 files at a time.
    request_id: optional
```

Response:
```python
{
    request_id: <request_id>,
    prediction: [
                {
                    filename: <filename>,
                    objects: [
                        CAR, PEOPLE, DOG, ...
                    ],
                    describe: {
                        type: <'exterior', 'interior', 'engine'>
                        view: <'e_door_handle', 'e_front_view', 'e_grille', 'e_headlight', 'e_left_front_view', 'e_left_rear_view',
                            'e_left_view', 'e_rear_light', 'e_rear_view', 'e_right_front_view', 'e_right_rear_view', 'e_right_view',
                            'e_side_mirror', 'e_sun_roof', 'e_top_view', 'e_wheel',
                            'i_rear_view_mirror', 'i_driving_seat_side_view', 'i_full_interior_view', 'i_gear_stick', 'i_media_screen',
                            'i_odometer', 'i_rear_seat_front_view', 'i_rear_seat_side_view', 'i_steering_wheel',
                            'i_steering_wheel_dash_board', 'i_sun_roof_view', 'i_trunk', 'i_button', 'i_door_handle'>
                        marks: [
                            (CAR, (x1,y1,x2,y2)),
                            (PEOPLE, (x1,y1,x2,y2)),
                            ...
                        ]
                    }
                },
                ...
            ]
}
```
