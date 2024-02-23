import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

function createScene (containterId, objPath) {

    const container = document.getElementById(containterId);
    container.style.position = 'relative';

    if (containterId != "container1") {
        container.style.width = '15%';
    }
    container.style.display = 'inline-block';
    let renderer, stats, gui;
    let scene, camera, controls, cube, dirlight, ambientLight;
    let isinitialized = false;

    function initScene () {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);
        camera = new THREE.PerspectiveCamera(75, container.clientWidth / (window.innerHeight * 0.5), 0.1, 1000);

        renderer = new THREE.WebGLRenderer();
        renderer.setSize(container.clientWidth, window.innerHeight * 0.5);
        container.appendChild(renderer.domElement);

        controls = new OrbitControls(camera, renderer.domElement);
        controls.minDistance = 2;
        controls.maxDistance = 10;
        controls.addEventListener('change', function () { renderer.render(scene, camera); });

        dirlight = new THREE.DirectionalLight(0xffffff, 0.5);
        dirlight.position.set(0, 0, 1);
        scene.add(dirlight);

        ambientLight = new THREE.AmbientLight(0x404040, 2);
        scene.add(ambientLight);


        // the loading of the object is asynchronous
        let loader = new OBJLoader();
        loader.load(
            // resource URL
            objPath,
            // called when resource is loaded
            function (object) {
                cube = object.children[0];
                cube.material = new THREE.MeshPhongMaterial({ color: 0x999999 });
                cube.position.set(0, 0, 0);
                cube.scale.set(15.0, 15.0, 15.0);
                cube.name = "cube";
                scene.add(cube);
                initGUI(); // initialize the GUI after the object is loaded
            },
            // called when loading is in progresses
            function (xhr) {
                console.log((xhr.loaded / xhr.total * 100) + '% loaded');
            },
            // called when loading has errors
            function (error) {
                console.log('An error happened' + error);
            }
        );

        camera.position.z = 5;
    }


    function initGUI () {
        if (!isinitialized && cube) { // Ensure the cube has been loaded
            // gui = new GUI();
            // gui.add(cube.position, 'x', -1, 1);
            // gui.add(cube.position, 'y', -1, 1);
            // gui.add(cube.position, 'z', -1, 1);
            // gui.domElement.style.position = 'relative'; // Position relative to the container
            // gui.domElement.style.width = '100%'; // GUI spans the width of the container
            // // Use transform to scale down the size of the GUI
            // gui.domElement.style.transform = 'scale(0.75)'; // Adjust the scale factor as needed
            // gui.domElement.style.transformOrigin = 'top'; // Ensures the scaling doesn't affect the position
            // container.appendChild(gui.domElement);
            isinitialized = true;
        }
    }



    function animate () {
        requestAnimationFrame(animate);

        cube = scene.getObjectByName("cube");
        if (cube) {
            cube.rotation.x += 0.01;
            cube.rotation.y += 0.01;
        }

        renderer.render(scene, camera);
    }

    function onWindowResize () {
        camera.aspect = container.clientWidth / (window.innerHeight * 0.5);
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, window.innerHeight * 0.5);
    };

    window.addEventListener('resize', onWindowResize, false);

    initScene();
    animate();

}

createScene('container0', 'assets/bunny.obj');
createScene('container1', 'assets/bunny_subdivided.obj');
createScene('container2', 'assets/bunny.obj');
createScene('container3', 'assets/bunny_subdivided.obj');
createScene('container4', 'assets/bunny_subdivided_10000.obj');
createScene('container5', 'assets/bunny_subdivided_1000.obj');
createScene('container6', 'assets/bunny_subdivided_500.obj');
createScene('container7', 'assets/bunny_subdivided_200.obj');