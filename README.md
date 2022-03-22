<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">

  <!--
  <script src="./resources/jsapi" type="text/javascript"></script>
  <script type="text/javascript" async>google.load("jquery", "1.3.2");</script>
 -->

<style type="text/css">
  body {
    font-family: "HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 1100px;
    text-align: justify;
  }
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }
  .disclaimerbox {
    background-color: #eee;
    border: 1px solid #eeeeee;
    border-radius: 10px ;
    -moz-border-radius: 10px ;
    -webkit-border-radius: 10px ;
    padding: 20px;
  }
  video.header-vid {
    height: 140px;
    border: 1px solid black;
    border-radius: 10px ;
    -moz-border-radius: 10px ;
    -webkit-border-radius: 10px ;
  }
  img.header-img {
    height: 140px;
    border: 1px solid black;
    border-radius: 10px ;
    -moz-border-radius: 10px ;
    -webkit-border-radius: 10px ;
  }
  img.rounded {
    border: 0px solid #eeeeee;
    border-radius: 10px ;
    -moz-border-radius: 10px ;
    -webkit-border-radius: 10px ;
  }
  a:link,a:visited
  {
    color: #1367a7;
    text-decoration: none;
  }
  a:hover {
    color: #208799;
  }
  td.dl-link {
    height: 160px;
    text-align: center;
    font-size: 22px;
  }
  .layered-paper-big { /* modified from: http://css-tricks.com/snippets/css/layered-paper/ */
    box-shadow:
            0px 0px 1px 1px rgba(0,0,0,0.35), /* The top layer shadow */
            5px 5px 0 0px #fff, /* The second layer */
            5px 5px 1px 1px rgba(0,0,0,0.35), /* The second layer shadow */
            10px 10px 0 0px #fff, /* The third layer */
            10px 10px 1px 1px rgba(0,0,0,0.35), /* The third layer shadow */
            15px 15px 0 0px #fff, /* The fourth layer */
            15px 15px 1px 1px rgba(0,0,0,0.35), /* The fourth layer shadow */
            20px 20px 0 0px #fff, /* The fifth layer */
            20px 20px 1px 1px rgba(0,0,0,0.35), /* The fifth layer shadow */
            25px 25px 0 0px #fff, /* The fifth layer */
            25px 25px 1px 1px rgba(0,0,0,0.35); /* The fifth layer shadow */
    margin-left: 10px;
    margin-right: 45px;
  }
  .layered-paper { /* modified from: http://css-tricks.com/snippets/css/layered-paper/ */
    box-shadow:
            0px 0px 1px 1px rgba(0,0,0,0.35), /* The top layer shadow */
            5px 5px 0 0px #fff, /* The second layer */
            5px 5px 1px 1px rgba(0,0,0,0.35), /* The second layer shadow */
            10px 10px 0 0px #fff, /* The third layer */
            10px 10px 1px 1px rgba(0,0,0,0.35); /* The third layer shadow */
    margin-top: 5px;
    margin-left: 10px;
    margin-right: 30px;
    margin-bottom: 5px;
  }
  .vert-cent {
    position: relative;
      top: 50%;
      transform: translateY(-50%);
  }
  hr
  {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }
</style>



    <title>TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing    </title>
    <meta property="og:title" content="TransEditor">
  </head>

  <body>
        <br>
          <center>
            <span style="font-size:34px">TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing</span><br><br>

          <table align="center" width="850px">
            <tbody><tr>
                    <td align="center" width="205px">
              <center>
                <span style="font-size:20px"><a href="https://github.com/BillyXYB">Yanbo Xu*</a></span>
                </center>
                </td>
                    <td align="center" width="175px">
              <center>
                <span style="font-size:20px"><a href="https://github.com/yinyueqin">Yueqin Yin*</a></span>
                </center>
                </td>
                    <td align="center" width="175px">
              <center>
                <span style="font-size:20px"><a href="https://liming-jiang.com/">Liming Jiang</a></span>
                </center>
                </td>
                    <td align="center" width="175px">
              <center>
                <span style="font-size:20px"><a href="https://qianyiwu.github.io/">Qianyi Wu</a></span>
                </center>
                </td>
            </tr>
        </tbody></table>


        <table align="center" width="850px">
            <tbody><tr>
                <td align="center" width="175px">
                    <center>
                      <span style="font-size:20px"><a href="https://github.com/daili0015">Chengyao Zheng</a></span>
                      </center>
                      </td>
                <td align="center" width="175px">
                <center>
                    <span style="font-size:20px"><a href="https://www.mmlab-ntu.com/person/ccloy/">Chen Change Loy</a></span>
                    </center>
                    </td>
                <td align="center" width="175px">
                    <center>
                        <span style="font-size:20px"><a href="http://daibo.info/">Bo Dai</a></span>
                        </center>
                        </td>
                <td align="center" width="175px">
                    <center>
                        <span style="font-size:20px"><a href="https://dblp.org/pid/50/8731.html">Wayne Wu</a><sup>+</sup></span>
                        </center>
                        </td>
            </tr>
        </tbody></table>


        <table align=center width=900px>
            <tr>
            <td align=center width=80px>
            <center>
            <span style="font-size:20px">Shanghai AI Laboratory</span>
            </center>
            </td>
            <td align=center width=80px>
            <center>
            <span style="font-size:20px">Nanyang Technological University</span>
            </center>
            </td>
            <td align=center width=80px>
            <center>
            <span style="font-size:20px">SenseTime Research</span>
            </center>
            </td>
             <td align=center width=80px>
             <center>
             <span style="font-size:20px">HKUST</span>
             </center>
             </td>
     
             <td align=center width=80px>
             <center>
             <span style="font-size:20px">Monash University</span>
             </center>
             </td>
          </tr>
         </table>


          <table align="center" width="700px">
            <tbody><tr>
<!--                     <td align="center" width="50px">
              <center>
                    <span style="font-size:18px"></span>
                </center>
                </td> -->
                    <td align="center" width="200px">
              <center>
                <br>
                <span style="font-size:20px">Code <a href="https://github.com/BillyXYB/TransEditor"> [GitHub]</a></span>
                </center>
                </td>
                    <td align="center" width="200px">
              <center>
                <br>
                <span style="font-size:20px">CVPR 2022 <a href="."> <!-- [Paper] -->[Paper]</a></span>
                </center>
                </td>

        </tr></tbody></table>
          </center>
        
          <!-- <center><h2>Overview</h2></center>


      <p align="center">Our Method shows its effectiveness for highly controllable facial editing. </p>-->
      
      <center><img src="files/teaser.png" width="900px"/></center>
        <center><h2>Abstract</h2></center>
        Recent advances like StyleGAN have promoted the growth of controllable facial editing. To address its core challenge of attribute decoupling in a single latent space, attempts have been made to adopt dual-space GAN for better disentanglement of style and content representations. Nonetheless, these methods are still incompetent to obtain plausible editing results with high controllability, especially for complicated attributes. In this study, we highlight the importance of interaction in a dual-space GAN for more controllable editing. We propose TransEditor, a novel Transformer-based framework to enhance such interaction. Besides, we develop a new dual-space editing and inversion strategy to provide additional editing flexibility. Extensive experiments demonstrate the superiority of the proposed framework in image quality and editing capability, suggesting the effectiveness of TransEditor for highly controllable facial editing.<br>

      
      <br><hr>
        <center><h2>Paper</h2></center><table align="center" width="700" px="">

          <tbody><tr>
          <td><a href="."><img class="layered-paper-big" style="height:175px" src="./files/firstpg.png"></a></td>
          <td><span style="font-size:12pt">Yanbo Xu*, Yueqin Yin*, Liming Jiang, Qianyi Wu, Chengyao Zheng, Chen Change Loy, Bo Dai, Wayne Wu.</span><br>
          <b><span style="font-size:12pt">TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing .</span></b><br>
          <span style="font-size:12pt">In CVPR, 2022. (<a href=".">Paper</a>)</span>
          </td>

          <br>
          <table align="center" width="600px">
            <tbody>
              <tr>
                <td>
                  <center>
                    <span style="font-size:22px">
                      <a href="./files/bibtex.txt" target="_blank">[Bibtex]</a>
                    </span>
                  </center>
                </td>
              </tr>
            </tbody>
          </table>
      <br>

        

      <br><hr>

       <center><h2>Method</h2></center>
       <p align="center">Two latent spaces Z and P are used for generation. We correlate them via a cross-attention-based interaction module to facilitate editing. </p>
       
       <center><img src="files/Model_Archi.png" width="1000px"/></center>

       <br><hr>

       <p align="center"><b>Interpolation of two latent spaces.</b> They are disentangled with different semantic meanings.</p>
       <table align="center" cellpadding="0" cellspacing="10" >
         <tr>
           <center><td  align="center">Interpolating Z space<br> <img src="files/interp_style_celeba.png"  width=480px></td></center>
           <center><td  align="center">Interpolating P space <br> <img src="files/interp_content_celeba.png" width=480px></td></center>
           
         </tr>
       </table>


    <br><hr>
    <center><h2>Editing Results</h2></center>
      
    <table align="center" cellpadding="0" cellspacing="10" >
      <tr>
        <center><td  align="center">Smile editing on Z space<br> <img src="files/edit_smile_celeba.png" width=460px border=1></td></center>
        <center><td  align="center">Gender editing on Z and P space <br> <img src="files/edit_gender_ffhq.png"  width=460px border=1></td></center>
      </tr>
    </table>
    
    <table align="center" cellpadding="0" cellspacing="10" >
      <tr>
        <center><td  align="center">Head pose editing on P space <br> <img src="files/edit_pose_ffhq.png"  height=350px border=1></td></center>
        <center><td  align="center">Age editing on Z and P space<br> <img src="files/edit_age_ffhq.png"  height=350px border=1></td></center>
      </tr>
    </table>
    <br><hr>

     <center><h2>Comparison</h2></center>
    <p align="center">Our method shows better editing ability compared with other SOTA methods.</p>
    
    <table align="center" cellpadding="0" cellspacing="10" >
      <tr>
        <center><td  align="center">Gender Editing Comparison <br> <img src="files/no_inverison_gender.png"  height=400px border=1></td></center>
        <center><td  align="center">Pose Editing Comparison<br> <img src="files/no_inversion_pose.png"  height=400px border=1></td></center>
      </tr>
    </table>

    <br><hr>

        <table align="center" width="1100px">
          <tbody><tr>
                  <td width="400px">
            <left>
          <center><h2>Acknowledgements</h2></center>
          This study is partly supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).
      </left>
    </td>
       </tr>
    </tbody></table>

    <br><br>
    <br><br>
    <br><br>
    

<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-5MWL7CQ4Z7"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-5MWL7CQ4Z7');
</script>


</body></html>
