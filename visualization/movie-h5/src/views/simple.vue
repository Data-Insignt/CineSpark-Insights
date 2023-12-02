<template>
    <div class="app-container">
        <div class="part">
            <chart-one />
        </div>
        <div class="part">
            <chart-two />
        </div>
        <div class="headers">
            <div class="item active">Simple Recommendation</div>
            <div class="item" @click="jump('/als')">ALS Model</div>
            <div class="item" @click="jump('/rf')">RF Model</div>
            <div class="item" @click="jump('/com')">Comparing</div>
        </div>
    </div>
</template>
<script>
import chartOne from './charts/chartOne'
import chartTwo from './charts/chartTwo'
export default {
    components: {chartOne, chartTwo},
    methods: {
        jump(url) {
            this.$router.push(url)
        }
    }
}
</script>
<style scoped>
.app-container {
    padding-bottom: 50px;
}
.part {
    background: #fff;
    padding: 10px 0;
    width: 95%;
    margin: 10px auto;
    border-radius: 5px;
}
.headers {
    display: flex;
    align-items: center;
    background: #ffff;
    color: #333;
    font-size: 14px;
    text-align: center;
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    height: 50px;
    border-top: 1px solid #eee;
}
.headers .item {
    flex: 1;
    padding: 5px;
}
.headers .item.active {
    color: #e63737;
}
</style>