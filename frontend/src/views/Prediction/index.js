import React, { useEffect, useState } from 'react';
import { Button, Container, Row, Col } from 'react-bootstrap'
import car1 from '../../images/carouselImages/car1.jpg'
import ProductCard from '../../components/productCard'
import axios from 'axios'

function Prediction(props) {
    const [file, setFile] = useState('')
    const [imageURL, setImageURL] = useState('')
    const [prediction, setPrediction] = useState('')
    const [display, setDisplay] = useState('')
    const [car, setCar] = useState()

    const uploadImage = (e) => {

        e.preventDefault()

        let formData = new FormData();

        formData.append('file', file);

        axios.post(`/predict`, formData)
            .then(res => {
                setCar(JSON.parse(res.data))
            })
    }

    const chooseFile = (e) => {
        const file = e.target.files[0]
        setFile(file)
        setImageURL(URL.createObjectURL(file))
    }

    const myStyle = {
        borderRight: 'solid 2px lightgrey',
        padding: "10px"
    };

    return (
        <Container>
            <Row>
                <Col lg="6" style={myStyle}>
                    <form encType="multipart/form-data" onSubmit={uploadImage}>
                        <input type='file' onChange={chooseFile} placeholder="Choosing image" />
                    </form>
                    <img src={imageURL} style={{ marginTop: '10px' }} width='330' /><br></br>
                    <Button variant="primary" onClick={uploadImage} disabled={!file} style={{ marginTop: '10px' }}>Tìm kiếm</Button>
                </Col>
                <Col lg="6">
                    <h4>Sản phẩm có thể bạn quan tâm:</h4>
                    {car && <Col><ProductCard name={car.name} price={car.price} imgId={car.imageId} /></Col>}
                </Col>
            </Row>
        </Container>
    )
}

export default Prediction;